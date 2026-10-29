"""Hand-written rules: the baseline most banks start with.

Rules are transparent and fast, but rigid: thresholds fixed by hand catch
obvious fraud and miss anything that adapts. We keep them for two reasons:
1. as a baseline the ML system has to beat in rupee terms, and
2. as hard guardrails (e.g. the UPI per-transaction limit) applied before
   the model in production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CostConfig
from .decision import APPROVE, BLOCK, STEP_UP


def hard_guardrail(txn: dict, cost: CostConfig) -> str | None:
    """Non-negotiable checks applied before any model. Returns a reason or None."""
    if float(txn["amount"]) <= 0:
        return "INVALID_AMOUNT"
    if float(txn["amount"]) > cost.upi_txn_limit:
        return "ABOVE_UPI_TXN_LIMIT"
    return None


def rules_baseline(feats: pd.DataFrame) -> np.ndarray:
    """Typical first-generation rule set, applied to the feature table."""
    f = feats
    block = (
        ((f.is_new_device == 1) & (f.amount > 10_000))
        | ((f.user_txn_count_1h >= 3) & (f.user_amount_sum_1h > 25_000))
    )
    step = (
        ((f.amount_vs_user_max > 3) & (f.amount > 5_000))
        | ((f.is_night == 1) & (f.is_new_payee == 1) & (f.amount > 5_000))
        | ((f.is_collect == 1) & (f.amount > 10_000))
    )
    return np.where(block, BLOCK, np.where(step, STEP_UP, APPROVE))
