"""Online scoring service: guardrails -> features -> model -> decision -> reasons.

Framework-agnostic so it can sit behind FastAPI (api.py), a Kafka consumer,
or a batch job unchanged.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import joblib
import pandas as pd

from .config import ARTIFACT_DIR, CostConfig
from .decision import APPROVE, BLOCK, Thresholds
from .features import FEATURE_NAMES, FeatureStore
from .rules import hard_guardrail


class FraudScorer:
    def __init__(self, model, store: FeatureStore, explainer, thresholds: Thresholds,
                 cost: CostConfig | None = None, metadata: dict | None = None) -> None:
        self.model = model
        self.store = store
        self.explainer = explainer
        self.thresholds = thresholds
        self.cost = cost or CostConfig()
        self.metadata = metadata or {}
        self._lock = threading.Lock()   # the feature store is stateful

    @classmethod
    def load(cls, artifact_dir: Path | str = ARTIFACT_DIR) -> "FraudScorer":
        d = Path(artifact_dir)
        meta = json.loads((d / "metadata.json").read_text())
        return cls(
            model=joblib.load(d / "model.joblib"),
            store=joblib.load(d / "feature_store.joblib"),
            explainer=joblib.load(d / "explainer.joblib"),
            thresholds=Thresholds(**meta["thresholds"]),
            cost=CostConfig(**meta["cost_config"]),
            metadata=meta,
        )

    def score(self, txn: dict) -> dict:
        start = time.perf_counter()
        result = {"txn_id": txn.get("txn_id"), "model_version": self.metadata.get("model_version")}

        guard = hard_guardrail(txn, self.cost)
        if guard:
            result.update(risk_score=None, decision=BLOCK,
                          reasons=[{"code": guard, "description": "Failed a hard business rule"}])
        else:
            with self._lock:
                feats = self.store.compute(txn)
                self.store.update(txn)   # record the attempt: velocity counts attempts too
            x = pd.DataFrame([feats], columns=FEATURE_NAMES)
            risk = float(self.model.predict_proba(x)[0, 1])
            decision = self.thresholds.decide(risk)
            reasons = [] if decision == APPROVE else self.explainer.explain(feats)
            result.update(risk_score=round(risk, 6), decision=decision, reasons=reasons)

        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result
