"""Run:  python -m unittest discover tests -v     (or: pytest)"""
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraudshield.config import ARTIFACT_DIR, CostConfig, SimulationConfig  # noqa: E402
from fraudshield.data_generator import generate_transactions  # noqa: E402
from fraudshield.decision import APPROVE, BLOCK, STEP_UP, Thresholds, optimise_thresholds, outcome_cost  # noqa: E402
from fraudshield.features import FeatureStore, build_feature_table  # noqa: E402
from fraudshield.monitoring import psi  # noqa: E402


def txn(ts, user="U1", payee="M1", amount=100.0, device="D1", kind="P2M"):
    return {"timestamp": ts, "user_id": user, "payee_id": payee, "amount": amount,
            "device_id": device, "txn_type": kind}


class FeatureStoreTests(unittest.TestCase):
    def test_features_use_only_past_information(self):
        fs = FeatureStore()
        f1 = fs.compute_and_update(txn(1_000))
        self.assertEqual(f1["user_history_count"], 0)       # does not see itself
        self.assertEqual(f1["user_txn_count_1h"], 0)
        f2 = fs.compute_and_update(txn(1_060, amount=500))
        self.assertEqual(f2["user_txn_count_1h"], 1)
        self.assertEqual(f2["user_amount_sum_1h"], 100.0)
        self.assertEqual(f2["secs_since_last_txn"], 60)

    def test_time_windows_expire(self):
        fs = FeatureStore()
        fs.compute_and_update(txn(0))
        f = fs.compute(txn(2 * 3600))                        # 2 hours later
        self.assertEqual(f["user_txn_count_1h"], 0)
        self.assertEqual(f["user_txn_count_24h"], 1)
        f = fs.compute(txn(25 * 3600))                       # >24 hours later
        self.assertEqual(f["user_txn_count_24h"], 0)

    def test_new_device_and_payee_flags(self):
        fs = FeatureStore()
        fs.compute_and_update(txn(0))
        same = fs.compute(txn(10))
        self.assertEqual((same["is_new_device"], same["is_new_payee"]), (0, 0))
        new = fs.compute(txn(20, payee="MULE", device="EVIL"))
        self.assertEqual((new["is_new_device"], new["is_new_payee"]), (1, 1))

    def test_payee_fan_in_counts_distinct_senders(self):
        fs = FeatureStore()
        for i, u in enumerate(["A", "B", "B", "C"]):
            fs.compute_and_update(txn(i, user=u, payee="P"))
        self.assertEqual(fs.compute(txn(10, user="D", payee="P"))["payee_distinct_senders_24h"], 3)

    def test_rejects_out_of_order_events(self):
        fs = FeatureStore()
        fs.update(txn(100))
        with self.assertRaises(ValueError):
            fs.update(txn(50))

    def test_offline_and_online_features_match(self):
        """Training/serving parity: batch replay == one-by-one live scoring."""
        df = generate_transactions(SimulationConfig(n_users=60, n_merchants=30,
                                                    n_mule_accounts=4, days=10, seed=7))
        offline, _ = build_feature_table(df)
        live = FeatureStore()
        for i, row in enumerate(df.to_dict("records")):
            row["timestamp"] = row["timestamp"].isoformat()
            online = live.compute_and_update(row)
            np.testing.assert_allclose(offline.iloc[i].to_numpy(),
                                       pd.Series(online)[offline.columns].to_numpy(), rtol=1e-9)


class DecisionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = 400
        self.y = rng.random(self.n) < 0.05
        self.scores = np.clip(rng.normal(0.2, 0.1, self.n) + 0.5 * self.y, 0, 1)
        self.amount = rng.uniform(100, 20_000, self.n)
        self.cost = CostConfig(max_block_rate=0.05)

    def test_outcome_cost_arithmetic(self):
        c = CostConfig(step_up_friction_cost=10, step_up_catch_rate=0.5,
                       block_legit_cost=100, review_cost_per_block=5)
        r = outcome_cost([APPROVE, STEP_UP, STEP_UP, BLOCK, BLOCK],
                         [1, 1, 0, 0, 1], [1000, 400, 50, 50, 300], c)
        # 1000 missed + 0.5*400 + 10 friction + 100 wrong block + 2*5 review
        self.assertAlmostEqual(r["total_cost"], 1000 + 200 + 10 + 100 + 10)

    def test_optimiser_matches_brute_force(self):
        th = optimise_thresholds(self.scores, self.y, self.amount, self.cost, n_candidates=60)
        best_opt = outcome_cost([th.decide(s) for s in self.scores], self.y, self.amount, self.cost)
        grid = np.unique(np.concatenate([self.scores, [np.inf]]))
        best = np.inf
        for lo, hi in itertools.product(grid[-60:], grid[-60:]):
            if lo > hi:
                continue
            acts = [Thresholds(lo, hi).decide(s) for s in self.scores]
            if np.mean(np.array(acts) == BLOCK) > self.cost.max_block_rate:
                continue
            best = min(best, outcome_cost(acts, self.y, self.amount, self.cost)["total_cost"])
        self.assertLessEqual(best_opt["total_cost"], best * 1.02)

    def test_optimiser_respects_block_capacity(self):
        th = optimise_thresholds(self.scores, self.y, self.amount, self.cost)
        block_rate = np.mean([th.decide(s) == BLOCK for s in self.scores])
        self.assertLessEqual(block_rate, self.cost.max_block_rate)


class MonitoringTests(unittest.TestCase):
    def test_psi(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0, 1, 20_000)
        self.assertLess(psi(a, rng.normal(0, 1, 20_000)), 0.01)
        self.assertGreater(psi(a, rng.normal(1.0, 1, 20_000)), 0.25)


@unittest.skipUnless((ARTIFACT_DIR / "model.joblib").exists(), "run run_pipeline.py first")
class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fraudshield.service import FraudScorer
        cls.scorer = FraudScorer.load()
        cls.ts = pd.Timestamp(cls.scorer.store.last_ts, unit="s") + pd.Timedelta(days=1)

    def test_guardrail_blocks_above_upi_limit(self):
        r = self.scorer.score({**txn(self.ts.isoformat(), user="U00001", amount=2_00_000),
                               "txn_id": "X1"})
        self.assertEqual(r["decision"], BLOCK)
        self.assertEqual(r["reasons"][0]["code"], "ABOVE_UPI_TXN_LIMIT")

    def test_score_shape(self):
        r = self.scorer.score({**txn(self.ts.isoformat(), user="U00002"), "txn_id": "X2"})
        self.assertIn(r["decision"], {APPROVE, STEP_UP, BLOCK})
        self.assertTrue(0 <= r["risk_score"] <= 1)
        self.assertLess(r["latency_ms"], 200)


try:
    import fastapi  # noqa: F401
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@unittest.skipUnless(HAS_FASTAPI and (ARTIFACT_DIR / "model.joblib").exists(),
                     "needs fastapi + trained artifacts")
class ApiTests(unittest.TestCase):
    def test_score_endpoint(self):
        from fastapi.testclient import TestClient
        from fraudshield.api import app
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").json()["status"], "ok")
            info = client.get("/v1/model-info").json()
            ts = pd.Timestamp(9e9, unit="s").isoformat()        # far future, always in order
            body = {"txn_id": "API1", "timestamp": ts, "user_id": "U00003",
                    "payee_id": "M00001", "amount": 250, "txn_type": "P2M", "device_id": "D00003a"}
            r = client.post("/v1/score", json=body)
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["model_version"], info["model_version"])
            bad = client.post("/v1/score", json={**body, "amount": -5})
            self.assertEqual(bad.status_code, 422)


if __name__ == "__main__":
    unittest.main()
