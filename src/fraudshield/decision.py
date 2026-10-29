"""Turn a fraud score into a business decision: APPROVE, STEP_UP or BLOCK.

Accuracy is the wrong target for fraud (approving everything is 99.8%
accurate). What the business cares about is *rupees lost*:

    missed fraud        -> lose the transaction amount
    needless step-up    -> small friction cost (some customers abandon)
    wrongly blocked     -> large cost (angry customer, support call, churn)

We pick two score thresholds on a validation period that minimise the
total expected cost, subject to an operations limit on how many payments
may be blocked. Thresholds are then *frozen* and evaluated on unseen data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import CostConfig

APPROVE, STEP_UP, BLOCK = "APPROVE", "STEP_UP", "BLOCK"


@dataclass(frozen=True)
class Thresholds:
    step_up: float
    block: float

    def decide(self, score: float) -> str:
        if score >= self.block:
            return BLOCK
        if score >= self.step_up:
            return STEP_UP
        return APPROVE

    def to_dict(self) -> dict:
        return asdict(self)


def outcome_cost(actions, y, amount, cost: CostConfig) -> dict:
    """Total rupee cost of a set of decisions, broken down by source."""
    actions, y, amount = np.asarray(actions), np.asarray(y).astype(bool), np.asarray(amount, float)
    appr, step, blk = actions == APPROVE, actions == STEP_UP, actions == BLOCK
    parts = {
        "missed_fraud_loss": float(amount[appr & y].sum()),
        "fraud_loss_after_step_up": float((1 - cost.step_up_catch_rate) * amount[step & y].sum()),
        "step_up_friction": float(cost.step_up_friction_cost * (step & ~y).sum()),
        "wrong_block_cost": float(cost.block_legit_cost * (blk & ~y).sum()),
        "review_cost": float(cost.review_cost_per_block * blk.sum()),
    }
    total_fraud = float(amount[y].sum())
    prevented = total_fraud - parts["missed_fraud_loss"] - parts["fraud_loss_after_step_up"]
    return {
        "total_cost": sum(parts.values()),
        **parts,
        "fraud_amount_total": total_fraud,
        "fraud_amount_prevented": prevented,
        "fraud_amount_prevented_pct": 100 * prevented / total_fraud if total_fraud else 0.0,
        "fraud_txn_recall_pct": 100 * float((~appr & y).sum()) / max(int(y.sum()), 1),
        "step_up_rate_pct": 100 * float(step.mean()),
        "block_rate_pct": 100 * float(blk.mean()),
        "legit_customers_challenged": int(((step | blk) & ~y).sum()),
    }


def optimise_thresholds(scores, y, amount, cost: CostConfig, n_candidates: int = 400) -> Thresholds:
    """Grid-search (step_up, block) thresholds minimising total cost.

    Uses prefix sums over score-sorted transactions, so evaluating each of
    the ~n_candidates^2 threshold pairs is O(1).
    """
    scores = np.asarray(scores, float)
    y = np.asarray(y).astype(bool)
    amount = np.asarray(amount, float)

    order = np.argsort(-scores, kind="stable")
    s_sorted = scores[order]
    fraud_amt = np.concatenate([[0.0], np.cumsum(np.where(y[order], amount[order], 0.0))])
    n_fraud = np.concatenate([[0], np.cumsum(y[order])])
    n_legit = np.concatenate([[0], np.cumsum(~y[order])])
    total_fraud_amt = fraud_amt[-1]
    n = len(scores)

    # Candidate thresholds concentrated in the extreme upper tail of scores,
    # where the decisions actually happen.
    tail = 1 - np.logspace(-5, np.log10(0.3), n_candidates)
    cand = np.unique(np.concatenate([np.quantile(scores, tail), [np.inf]]))
    # k(t) = number of transactions with score >= t (scores sorted descending)
    k = np.searchsorted(-s_sorted, -cand, side="right")

    c = cost
    best = (np.inf, np.inf, np.inf)
    for i_b, kb in enumerate(k):
        if kb / n > c.max_block_rate:
            continue
        block_cost = (c.block_legit_cost + c.review_cost_per_block) * n_legit[kb] \
            + c.review_cost_per_block * n_fraud[kb]
        # step-up region = (kb, ks]; only thresholds <= block threshold
        ks = k[: i_b + 1]
        total = (total_fraud_amt - fraud_amt[ks]
                 + (1 - c.step_up_catch_rate) * (fraud_amt[ks] - fraud_amt[kb])
                 + c.step_up_friction_cost * (n_legit[ks] - n_legit[kb])
                 + block_cost)
        j = int(np.argmin(total))
        if total[j] < best[0]:
            best = (float(total[j]), float(cand[j]), float(cand[i_b]))
    return Thresholds(step_up=best[1], block=best[2])
