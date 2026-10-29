"""End-to-end training pipeline.

    python -m fraudshield.pipeline            (from the src/ directory, or
    python run_pipeline.py                     from the project root)

Steps
  1. Generate 60 days of UPI transactions            -> data/transactions.csv.gz
  2. Build point-in-time features (no leakage)
  3. Time split: train (days 0-39) | valid (40-49) | test (50-59)
  4. Train rules baseline, logistic regression, gradient boosting
  5. Tune APPROVE / STEP_UP / BLOCK thresholds on VALID by rupee cost
  6. Report everything on the untouched TEST period
  7. Drift check (PSI) train vs test
  8. Save artifacts for the API                       -> artifacts/
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, DATA_DIR, REPORT_DIR, SETTINGS, Settings
from .data_generator import generate_transactions
from .decision import APPROVE, optimise_thresholds, outcome_cost
from .explain import ReasonExplainer
from .features import FEATURE_NAMES, build_feature_table
from .model import ranking_metrics, score, time_split, train_gbm, train_logistic
from .monitoring import drift_report
from .rules import rules_baseline


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(settings: Settings = SETTINGS) -> dict:
    for d in (DATA_DIR, ARTIFACT_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    _log("1/8 Generating transactions")
    df = generate_transactions(settings.sim)
    df.to_csv(DATA_DIR / "transactions.csv.gz", index=False)
    y = df["is_fraud"].to_numpy()
    _log(f"    {len(df):,} transactions, fraud rate {100 * y.mean():.3f}%")

    _log("2/8 Building point-in-time features")
    X, store = build_feature_table(df)

    _log("3/8 Time-based split")
    m = time_split(df, settings.split)
    split_info = {k: {"rows": int(v.sum()), "fraud": int(y[v].sum())} for k, v in m.items()}

    _log("4/8 Training models")
    lr = train_logistic(X[m["train"]], y[m["train"]])
    gbm = train_gbm(X[m["train"]], y[m["train"]], seed=settings.sim.seed)

    test_scores = {"logistic_regression": score(lr, X[m["test"]]),
                   "gradient_boosting": score(gbm, X[m["test"]])}
    ranking = {name: ranking_metrics(y[m["test"]], s) for name, s in test_scores.items()}

    _log("5/8 Optimising decision thresholds on validation period (by rupee cost)")
    thresholds = optimise_thresholds(score(gbm, X[m["valid"]]), y[m["valid"]],
                                     df.loc[m["valid"], "amount"], settings.cost)
    _log(f"    step-up >= {thresholds.step_up:.4f}, block >= {thresholds.block:.4f}")

    _log("6/8 Evaluating on unseen test period")
    y_te, amt_te = y[m["test"]], df.loc[m["test"], "amount"].to_numpy()
    model_actions = np.array([thresholds.decide(s) for s in test_scores["gradient_boosting"]])
    strategies = {
        "no_fraud_checks": np.full(len(y_te), APPROVE),
        "rules_only": rules_baseline(X[m["test"]]),
        "ml_cost_optimised": model_actions,
    }
    business = {k: outcome_cost(a, y_te, amt_te, settings.cost) for k, a in strategies.items()}
    ftype = df.loc[m["test"], "fraud_type"].to_numpy()
    by_pattern = {
        t: {"fraud_txns": int((ftype == t).sum()),
            "caught_pct": round(100 * float((model_actions[ftype == t] != APPROVE).mean()), 1)}
        for t in ("account_takeover", "scam")
    }

    _log("7/8 Drift check (PSI) train vs test")
    drift = drift_report(X.loc[m["train"], FEATURE_NAMES], X.loc[m["test"], FEATURE_NAMES],
                         score(gbm, X[m["train"]]), test_scores["gradient_boosting"])

    _log("8/8 Saving artifacts")
    explainer = ReasonExplainer.fit(gbm, X.loc[m["train"] & (y == 0)])
    version = datetime.now(timezone.utc).strftime("gbm-%Y%m%d-%H%M%S")
    joblib.dump(gbm, ARTIFACT_DIR / "model.joblib")
    joblib.dump(explainer, ARTIFACT_DIR / "explainer.joblib")
    # The store has replayed ALL history, so the API starts "warm".
    joblib.dump(store, ARTIFACT_DIR / "feature_store.joblib")
    metadata = {
        "model_version": version,
        "features": FEATURE_NAMES,
        "thresholds": thresholds.to_dict(),
        "cost_config": asdict(settings.cost),
        "trained_on_days": settings.split.train_days,
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    results = {"model_version": version, "split": split_info, "ranking_metrics_test": ranking,
               "thresholds": thresholds.to_dict(), "business_impact_test": business,
               "detection_by_fraud_pattern": by_pattern,
               "drift_alerts": drift[drift.status != "stable"].to_dict("records")}
    (REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    drift.to_csv(REPORT_DIR / "drift_report.csv", index=False)
    _write_report(results, drift)
    _plot_costs(business)
    _log(f"Done. Report: {REPORT_DIR / 'report.md'}")
    return results


def _inr(x: float) -> str:
    return f"Rs {x:,.0f}"


def _write_report(r: dict, drift: pd.DataFrame) -> None:
    b = r["business_impact_test"]
    base, rules, ml = b["no_fraud_checks"], b["rules_only"], b["ml_cost_optimised"]
    saving = rules["total_cost"] - ml["total_cost"]
    lines = [
        "# UPI Fraud Shield - results on unseen test period", "",
        f"Model version `{r['model_version']}`. Test period: {r['split']['test']['rows']:,} "
        f"transactions, {r['split']['test']['fraud']} fraudulent.", "",
        "## Business impact", "",
        "| Strategy | Total cost | Fraud prevented | Fraud txns caught | Legit customers challenged | Block rate |",
        "|---|---|---|---|---|---|",
    ]
    for name, v in b.items():
        lines.append(f"| {name} | {_inr(v['total_cost'])} | {v['fraud_amount_prevented_pct']:.1f}% | "
                     f"{v['fraud_txn_recall_pct']:.1f}% | {v['legit_customers_challenged']:,} | "
                     f"{v['block_rate_pct']:.2f}% |")
    lines += [
        "",
        f"**ML vs rules:** total cost down {_inr(saving)} "
        f"({100 * saving / rules['total_cost']:.0f}%) over {r['split']['test']['rows']:,} payments.",
        "",
        "## Detection by fraud pattern (ML)", "",
        "| Pattern | Fraud txns | Caught (step-up or block) |", "|---|---|---|",
        *[f"| {k} | {v['fraud_txns']} | {v['caught_pct']}% |"
          for k, v in r["detection_by_fraud_pattern"].items()],
        "",
        "## Model ranking quality (test)", "",
        "| Model | PR-AUC | ROC-AUC | Recall @ top 0.5% |", "|---|---|---|---|",
        *[f"| {k} | {v['pr_auc']:.3f} | {v['roc_auc']:.3f} | {100 * v['recall_at_top_0.5pct']:.1f}% |"
          for k, v in r["ranking_metrics_test"].items()],
        "",
        "## Cost breakdown (ML)", "",
        *[f"- {k.replace('_', ' ')}: {_inr(ml[k])}" for k in
          ("missed_fraud_loss", "fraud_loss_after_step_up", "step_up_friction",
           "wrong_block_cost", "review_cost")],
        "",
        "## Drift (PSI, train vs test) - top 5", "",
        "| Feature | PSI | Status |", "|---|---|---|",
        *[f"| {row.feature} | {row.psi:.3f} | {row.status} |" for row in drift.head(5).itertuples()],
        "",
    ]
    (REPORT_DIR / "report.md").write_text("\n".join(lines))


def _plot_costs(business: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"no_fraud_checks": "No fraud checks", "rules_only": "Rules only",
              "ml_cost_optimised": "ML + cost-optimised\nthresholds"}
    names = list(business)
    vals = [business[n]["total_cost"] / 1e5 for n in names]       # in lakh rupees
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.barh([labels[n] for n in names], vals, color="#2a78d6", height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"Rs {v:.1f} lakh", va="center", color="#0b0b0b", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_title("Total cost of fraud + friction, test period (10 days)",
                 loc="left", color="#0b0b0b", fontsize=11)
    ax.set_xlabel("Rs lakh", color="#52514e", fontsize=9)
    ax.tick_params(colors="#52514e", labelsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d6d5d0")
    ax.grid(axis="x", color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "cost_comparison.png", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    run()
