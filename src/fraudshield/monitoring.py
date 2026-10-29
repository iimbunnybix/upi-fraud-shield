"""Drift monitoring with the Population Stability Index (PSI).

Fraudsters adapt and customer behaviour shifts (festivals, salary days), so
a model silently decays. PSI compares each feature's distribution today with
the training period. Common rule of thumb:
    PSI < 0.10  stable | 0.10 - 0.25  watch | > 0.25  investigate / retrain
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected, actual, bins: int = 10) -> float:
    expected = np.asarray(expected, float)
    actual = np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:                       # constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    return "stable" if value < 0.10 else ("watch" if value < 0.25 else "ALERT")


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 ref_scores=None, cur_scores=None) -> pd.DataFrame:
    rows = [{"feature": c, "psi": psi(reference[c], current[c])} for c in reference.columns]
    if ref_scores is not None and cur_scores is not None:
        rows.append({"feature": "MODEL_SCORE", "psi": psi(ref_scores, cur_scores)})
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["status"] = out["psi"].map(status)
    return out
