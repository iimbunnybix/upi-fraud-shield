"""Reason codes: *why* was this payment challenged?

Regulators, support agents and customers all need an answer to "why was my
payment held?". We compute model-faithful reason codes by occlusion: reset
one group of related features to a typical (legit-median) value, re-score,
and see how much the risk drops. The groups with the largest drop are the
reasons. Batched into one predict call, this costs about a millisecond.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES

REASON_GROUPS: dict[str, tuple[str, list[str]]] = {
    "NEW_DEVICE": ("Payment from a device never seen on this account",
                   ["is_new_device", "user_device_count"]),
    "AMOUNT_SPIKE": ("Amount far above this customer's usual spending",
                     ["amount", "log_amount", "amount_zscore", "amount_vs_user_max"]),
    "PAYMENT_BURST": ("Unusually many or large payments in a short time",
                      ["user_txn_count_1h", "user_txn_count_24h", "user_amount_sum_1h",
                       "user_amount_sum_24h", "secs_since_last_txn"]),
    "NEW_PAYEE": ("First payment to this payee / many new payees today",
                  ["is_new_payee", "user_new_payees_24h"]),
    "UNUSUAL_HOUR": ("Payment at an unusual hour", ["hour", "is_night"]),
    "PAYMENT_TYPE": ("Collect request or person-to-person transfer", ["is_p2p", "is_collect"]),
    "PAYEE_RISK": ("Receiving account has a risky profile",
                   ["payee_distinct_senders_24h", "payee_age_days", "payee_inflow_per_day"]),
    "THIN_HISTORY": ("Little transaction history on this account", ["user_history_count"]),
}


class ReasonExplainer:
    def __init__(self, model, reference: pd.Series) -> None:
        self.model = model
        self.reference = reference[FEATURE_NAMES]

    @classmethod
    def fit(cls, model, X_legit: pd.DataFrame) -> "ReasonExplainer":
        return cls(model, X_legit[FEATURE_NAMES].median())

    def explain(self, features: dict, top_k: int = 3) -> list[dict]:
        base = pd.DataFrame([features], columns=FEATURE_NAMES)
        variants = [base]
        for _, cols in REASON_GROUPS.values():
            v = base.copy()
            v[cols] = self.reference[cols].to_numpy()
            variants.append(v)
        p = self.model.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
        drops = p[0] - p[1:]
        out = []
        for idx in np.argsort(-drops)[:top_k]:
            if drops[idx] <= 0.01 * max(p[0], 1e-9):
                break
            code = list(REASON_GROUPS)[idx]
            out.append({"code": code, "description": REASON_GROUPS[code][0],
                        "risk_contribution": round(float(drops[idx]), 4)})
        return out
