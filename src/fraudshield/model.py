"""Model training and ranking metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SplitConfig
from .features import FEATURE_NAMES


def time_split(df: pd.DataFrame, split: SplitConfig) -> dict[str, np.ndarray]:
    """Boolean masks for train / valid / test by calendar day (no shuffling)."""
    day = (df["timestamp"] - df["timestamp"].min()).dt.days.to_numpy()
    t, v = split.train_days, split.train_days + split.valid_days
    return {"train": (day >= split.burn_in_days) & (day < t),
            "valid": (day >= t) & (day < v), "test": day >= v}


def train_logistic(X: pd.DataFrame, y) -> object:
    """Simple, explainable baseline model."""
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
    return model.fit(X[FEATURE_NAMES], y)


def train_gbm(X: pd.DataFrame, y, seed: int = 0) -> HistGradientBoostingClassifier:
    """Gradient-boosted trees: captures interactions like
    'new device AND burst of payments AND unfamiliar payee'."""
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, class_weight={0: 1.0, 1: 20.0},
        early_stopping=False, random_state=seed,
    )
    return model.fit(X[FEATURE_NAMES], y)


def score(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[FEATURE_NAMES])[:, 1]


def ranking_metrics(y, s) -> dict:
    """PR-AUC is the headline metric for rare events; ROC-AUC flatters."""
    y = np.asarray(y)
    s = np.asarray(s)
    order = np.argsort(-s)
    out = {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s))}
    for pct in (0.1, 0.5, 1.0):
        k = max(int(len(s) * pct / 100), 1)
        out[f"recall_at_top_{pct}pct"] = float(y[order[:k]].sum() / max(y.sum(), 1))
    return out
