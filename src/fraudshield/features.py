"""Point-in-time feature store.

The single most common bug in fraud ML is *leakage*: computing a feature
with information that did not exist when the payment was made (e.g. a
"transactions in last hour" count that accidentally includes later ones).
Models trained that way look brilliant offline and fail in production.

This module avoids it by design:

* One `FeatureStore` class is used for BOTH offline training and the live
  API, so training/serving skew is impossible.
* For every transaction we first `compute()` features from the state as it
  was *before* the payment, then `update()` the state with it.
* Transactions must arrive in time order (as they do in a real stream).
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FeatureConfig

FEATURE_NAMES = [
    "amount", "log_amount", "hour", "is_night", "is_p2p", "is_collect",
    "user_history_count", "secs_since_last_txn",
    "user_txn_count_1h", "user_txn_count_24h",
    "user_amount_sum_1h", "user_amount_sum_24h",
    "amount_zscore", "amount_vs_user_max",
    "is_new_device", "user_device_count",
    "is_new_payee", "user_new_payees_24h",
    "payee_distinct_senders_24h", "payee_age_days", "payee_inflow_per_day",
]

_MAX_GAP_S = 30 * 86_400  # "never transacted before" is encoded as 30 days
# Cumulative quantities (history length, account age) grow forever and would
# make the model drift by construction. We cap them / turn them into rates so
# features mean the same thing on day 20 as on day 2,000.
_HISTORY_CAP = 20
_AGE_CAP_DAYS = 14.0


@dataclass
class _UserState:
    first_ts: float
    count: int = 0
    mean: float = 0.0           # Welford running mean of log(amount)
    m2: float = 0.0             # Welford running sum of squares
    max_amount: float = 0.0
    last_ts: float | None = None
    devices: set = field(default_factory=set)
    payees: set = field(default_factory=set)
    recent: deque = field(default_factory=deque)       # (ts, amount, was_new_payee)


@dataclass
class _PayeeState:
    inflow_count: int = 0
    recent: deque = field(default_factory=deque)       # (ts, sender)
    sender_counts: Counter = field(default_factory=Counter)


class FeatureStore:
    def __init__(self, cfg: FeatureConfig | None = None) -> None:
        self.cfg = cfg or FeatureConfig()
        self.users: dict[str, _UserState] = {}
        self.payees: dict[str, _PayeeState] = {}
        self.first_seen: dict[str, float] = {}         # any account, as sender or receiver
        self.last_ts: float = -math.inf

    # ------------------------------------------------------------------ utils
    def _expire(self, ts: float, user: _UserState | None, payee: _PayeeState | None) -> None:
        horizon = ts - self.cfg.long_window_s
        if user is not None:
            while user.recent and user.recent[0][0] < horizon:
                user.recent.popleft()
        if payee is not None:
            while payee.recent and payee.recent[0][0] < horizon:
                _, sender = payee.recent.popleft()
                payee.sender_counts[sender] -= 1
                if payee.sender_counts[sender] == 0:
                    del payee.sender_counts[sender]

    @staticmethod
    def _to_epoch(timestamp) -> float:
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
        return pd.Timestamp(timestamp).timestamp()

    # ---------------------------------------------------------------- compute
    def compute(self, txn: dict) -> dict:
        """Features for `txn` using only information available before it."""
        ts = self._to_epoch(txn["timestamp"])
        amount = float(txn["amount"])
        user = self.users.get(txn["user_id"])
        payee = self.payees.get(txn["payee_id"])
        self._expire(ts, user, payee)
        hour = int(ts // 3600) % 24     # timestamps are local (IST) wall-clock time

        f = {
            "amount": amount,
            "log_amount": math.log1p(amount),
            "hour": hour,
            "is_night": int(hour in self.cfg.night_hours),
            "is_p2p": int(txn["txn_type"] == "P2P"),
            "is_collect": int(txn["txn_type"] == "COLLECT"),
        }

        if user is None or user.count == 0:
            f.update(user_history_count=0, secs_since_last_txn=_MAX_GAP_S,
                     user_txn_count_1h=0, user_txn_count_24h=0,
                     user_amount_sum_1h=0.0, user_amount_sum_24h=0.0,
                     amount_zscore=0.0, amount_vs_user_max=1.0,
                     is_new_device=1, user_device_count=0,
                     is_new_payee=1, user_new_payees_24h=0)
        else:
            short_cut = ts - self.cfg.short_window_s
            c1 = s1 = 0
            s24 = 0.0
            new_payees_24h = 0
            for t, a, was_new in user.recent:
                s24 += a
                new_payees_24h += was_new
                if t >= short_cut:
                    c1 += 1
                    s1 += a
            std = math.sqrt(user.m2 / (user.count - 1)) if user.count > 1 else 1.0
            f.update(
                user_history_count=min(user.count, _HISTORY_CAP),
                secs_since_last_txn=min(ts - user.last_ts, _MAX_GAP_S),
                user_txn_count_1h=c1,
                user_txn_count_24h=len(user.recent),
                user_amount_sum_1h=s1,
                user_amount_sum_24h=s24,
                amount_zscore=(math.log1p(amount) - user.mean) / max(std, 0.25),
                amount_vs_user_max=amount / max(user.max_amount, 1.0),
                is_new_device=int(txn["device_id"] not in user.devices),
                user_device_count=len(user.devices),
                is_new_payee=int(txn["payee_id"] not in user.payees),
                user_new_payees_24h=new_payees_24h,
            )

        payee_first = self.first_seen.get(txn["payee_id"])
        age_days = 0.0 if payee_first is None else (ts - payee_first) / 86_400
        f["payee_distinct_senders_24h"] = len(payee.sender_counts) if payee else 0
        f["payee_age_days"] = min(age_days, _AGE_CAP_DAYS)
        f["payee_inflow_per_day"] = (payee.inflow_count / max(age_days, 1.0)) if payee else 0.0
        return f

    # ----------------------------------------------------------------- update
    def update(self, txn: dict) -> None:
        """Record `txn` so it is visible to features of later transactions."""
        ts = self._to_epoch(txn["timestamp"])
        if ts < self.last_ts:
            raise ValueError("Transactions must be processed in time order")
        self.last_ts = ts
        amount = float(txn["amount"])
        uid, pid = txn["user_id"], txn["payee_id"]

        self.first_seen.setdefault(uid, ts)
        self.first_seen.setdefault(pid, ts)

        user = self.users.get(uid)
        if user is None:
            user = self.users[uid] = _UserState(first_ts=ts)
        was_new_payee = int(pid not in user.payees)
        user.count += 1
        x = math.log1p(amount)
        delta = x - user.mean
        user.mean += delta / user.count
        user.m2 += delta * (x - user.mean)
        user.max_amount = max(user.max_amount, amount)
        user.last_ts = ts
        user.devices.add(txn["device_id"])
        user.payees.add(pid)
        user.recent.append((ts, amount, was_new_payee))

        payee = self.payees.get(pid)
        if payee is None:
            payee = self.payees[pid] = _PayeeState()
        payee.inflow_count += 1
        payee.recent.append((ts, uid))
        payee.sender_counts[uid] += 1

    def compute_and_update(self, txn: dict) -> dict:
        f = self.compute(txn)
        self.update(txn)
        return f


def build_feature_table(df: pd.DataFrame, store: FeatureStore | None = None) -> tuple[pd.DataFrame, FeatureStore]:
    """Replay a time-ordered transaction log through the store (offline)."""
    store = store or FeatureStore()
    df = df.sort_values("timestamp", kind="stable")
    records = df[["timestamp", "user_id", "payee_id", "amount", "txn_type", "device_id"]].copy()
    # epoch seconds, independent of pandas' datetime resolution (ns/us/s)
    records["timestamp"] = (records["timestamp"] - pd.Timestamp(0)) // pd.Timedelta(seconds=1)
    rows = [store.compute_and_update(r) for r in records.to_dict("records")]
    feats = pd.DataFrame(rows, index=df.index, columns=FEATURE_NAMES).astype(np.float64)
    return feats, store
