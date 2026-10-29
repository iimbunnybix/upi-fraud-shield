"""Central configuration: simulation size, time windows and business costs.

Every number that encodes a business assumption lives here so it can be
challenged in one place (and changed without touching model code).
"""
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class SimulationConfig:
    n_users: int = 4000
    n_merchants: int = 1500
    n_mule_accounts: int = 60          # accounts that receive stolen money
    days: int = 60
    start_date: str = "2026-07-01"
    # Fraud incidents per 1,000 users over the whole period, by pattern
    ato_incidents_per_1k: float = 30.0      # account takeover / SIM swap
    scam_incidents_per_1k: float = 40.0     # social engineering / collect-request scam
    seed: int = 42


@dataclass(frozen=True)
class SplitConfig:
    """Time-based split. Fraud models must be validated on the *future*."""
    burn_in_days: int = 14      # days 0-13  -> only warm up the feature store
                                #   (history-based features are unreliable while
                                #    every account still looks "brand new")
    train_days: int = 40        # days 14-39 -> fit the model
    valid_days: int = 10        # days 40-49 -> tune decision thresholds
    # remaining days            -> untouched test set (reported numbers)


@dataclass(frozen=True)
class CostConfig:
    """Rupee cost of each outcome. These drive the decision thresholds.

    approve + fraud   -> we lose the full transaction amount
    step_up + legit   -> friction: some customers abandon, support load
    step_up + fraud   -> step-up (re-auth with UPI PIN + device binding check)
                         stops only part of fraud: a scammer who has
                         already tricked the victim can still pass it
    block   + legit   -> angry customer, call-centre ticket, churn risk
    block   + fraud   -> fully prevented
    """
    step_up_friction_cost: float = 15.0
    step_up_catch_rate: float = 0.6
    block_legit_cost: float = 250.0
    review_cost_per_block: float = 40.0     # analyst time per blocked txn
    max_block_rate: float = 0.005           # ops capacity: block at most 0.5% of traffic
    upi_txn_limit: float = 1_00_000.0       # hard per-transaction limit (Rs 1 lakh)


@dataclass(frozen=True)
class FeatureConfig:
    short_window_s: int = 3600          # 1 hour
    long_window_s: int = 24 * 3600      # 24 hours
    night_hours: tuple = (0, 1, 2, 3, 4, 5)


@dataclass(frozen=True)
class Settings:
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)


SETTINGS = Settings()
