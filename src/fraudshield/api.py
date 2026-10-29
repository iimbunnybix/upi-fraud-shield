"""REST API for real-time scoring.

Run (from project root, after `python run_pipeline.py`):
    uvicorn fraudshield.api:app --app-dir src --port 8000
Docs at http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .service import FraudScorer

_scorer: FraudScorer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scorer
    _scorer = FraudScorer.load()          # load model + warm feature store once
    yield


app = FastAPI(title="UPI Fraud Shield", version="1.0", lifespan=lifespan)


class Transaction(BaseModel):
    txn_id: str = Field(..., examples=["T90000001"])
    timestamp: datetime = Field(..., examples=["2026-08-30T02:14:00"])
    user_id: str = Field(..., examples=["U00042"])
    payee_id: str = Field(..., examples=["U04012"])
    amount: float = Field(..., gt=0, examples=[24999.0])
    txn_type: Literal["P2P", "P2M", "COLLECT"] = "P2P"
    device_id: str = Field(..., examples=["F123456"])


class Reason(BaseModel):
    code: str
    description: str
    risk_contribution: float | None = None


class ScoreResponse(BaseModel):
    txn_id: str | None
    decision: Literal["APPROVE", "STEP_UP", "BLOCK"]
    risk_score: float | None
    reasons: list[Reason]
    model_version: str | None
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if _scorer else "loading"}


@app.get("/v1/model-info")
def model_info() -> dict:
    if not _scorer:
        raise HTTPException(503, "Model not loaded")
    m = _scorer.metadata
    return {"model_version": m["model_version"], "thresholds": m["thresholds"],
            "n_features": len(m["features"])}


@app.post("/v1/score", response_model=ScoreResponse)
def score(txn: Transaction) -> dict:
    if not _scorer:
        raise HTTPException(503, "Model not loaded")
    payload = txn.model_dump()
    payload["timestamp"] = payload["timestamp"].replace(tzinfo=None).isoformat()
    try:
        return _scorer.score(payload)
    except ValueError as e:                 # e.g. out-of-order timestamp
        raise HTTPException(422, str(e)) from e
