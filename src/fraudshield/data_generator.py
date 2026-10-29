"""Synthetic UPI transaction stream with realistic fraud patterns.

Real fraud data is confidential, so we simulate it. The goal is not to make
fraud easy to spot but to reproduce the *shape* of the real problem:

* extreme class imbalance (~0.3% fraud)
* legitimate behaviour that looks suspicious (new phone, first rent payment,
  night owls, occasional big purchases) -> hard negatives
* fraud that hides in normal-looking traffic:
    1. Account takeover (ATO): fraudster drains the account in a fast burst
       to mule accounts -- from a new device (SIM swap / stolen credentials)
       or from the victim's own phone via a remote-access app.
    2. Social-engineering scam: victim is tricked ("refund", "KYC update",
       collect request) into paying from *their own* phone. No device signal.
* mule accounts that look like ordinary low-activity users, run in short
  campaigns, and receive money from many unrelated victims.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SimulationConfig

DAY = 86_400


def _user_id(i: int) -> str:
    return f"U{i:05d}"


def _merchant_id(i: int) -> str:
    return f"M{i:05d}"


class _Builder:
    """Collects rows as column lists (much faster than appending dicts)."""

    cols = ("ts", "user_id", "payee_id", "amount", "txn_type",
            "device_id", "is_fraud", "fraud_type")

    def __init__(self) -> None:
        self.data = {c: [] for c in self.cols}

    def add(self, ts, user, payee, amount, txn_type, device, fraud_type=None):
        d = self.data
        d["ts"].append(int(ts))
        d["user_id"].append(user)
        d["payee_id"].append(payee)
        d["amount"].append(round(float(amount), 2))
        d["txn_type"].append(txn_type)
        d["device_id"].append(device)
        d["is_fraud"].append(int(fraud_type is not None))
        d["fraud_type"].append(fraud_type or "none")


def generate_transactions(cfg: SimulationConfig | None = None) -> pd.DataFrame:
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    horizon = cfg.days * DAY
    b = _Builder()

    n_users, n_merch = cfg.n_users, cfg.n_merchants
    merchants = [_merchant_id(i) for i in range(n_merch)]
    # Merchant popularity is heavy-tailed: a few big merchants get most traffic.
    merch_pop = rng.pareto(1.2, n_merch) + 1
    merch_pop /= merch_pop.sum()

    # Mules are ordinary-looking accounts placed in the same id space.
    mule_idx = np.arange(n_users, n_users + cfg.n_mule_accounts)
    all_user_ids = [_user_id(i) for i in range(n_users + cfg.n_mule_accounts)]

    # ---------------- per-user behavioural profile -----------------------
    typical_amt = np.exp(rng.normal(6.0, 0.8, n_users))        # median ~ Rs 400
    daily_rate = rng.gamma(2.0, 0.75, n_users)                  # mean 1.5 txn/day
    peak_hour = np.clip(rng.normal(15, 3, n_users), 7, 22)
    night_owl = rng.random(n_users) < 0.06
    pays_rent = rng.random(n_users) < 0.25
    group_settler = rng.random(n_users) < 0.08
    changes_phone_day = np.where(rng.random(n_users) < 0.03,
                                 rng.integers(5, cfg.days, n_users), -1)

    for u in range(n_users):
        uid = _user_id(u)
        fav_merch = rng.choice(n_merch, 6, p=merch_pop, replace=False)
        fav_p2p = rng.choice(n_users, 3, replace=False)
        device = f"D{u:05d}a"

        n_tx = rng.poisson(daily_rate[u] * cfg.days)
        ts = np.sort(rng.integers(0, cfg.days, n_tx)) * DAY
        hours = (rng.uniform(0, 24, n_tx) if night_owl[u]
                 else np.clip(rng.normal(peak_hour[u], 3, n_tx), 6, 23.9))
        ts = ts + (hours * 3600).astype(int)

        for t in ts:
            dev = device
            if 0 <= changes_phone_day[u] <= t // DAY:      # legit new phone
                dev = f"D{u:05d}b"
            r = rng.random()
            if r < 0.60:
                payee, kind = merchants[rng.choice(fav_merch)], "P2M"
            elif r < 0.80:
                payee, kind = merchants[rng.choice(n_merch, p=merch_pop)], "P2M"
            elif r < 0.93:
                payee, kind = _user_id(rng.choice(fav_p2p)), "P2P"
            else:
                payee, kind = _user_id(rng.integers(0, n_users)), "P2P"
            amt = typical_amt[u] * np.exp(rng.normal(0, 0.7))
            if rng.random() < 0.02:                         # occasional big buy
                amt *= rng.uniform(5, 20)
            if kind == "P2M" and rng.random() < 0.03:
                kind = "COLLECT"                            # legit collect requests
            b.add(t, uid, payee, max(amt, 1.0), kind, dev)

        if group_settler[u]:                                # legit burst: splitting a trip bill
            t = rng.integers(1, cfg.days) * DAY + int(rng.uniform(10, 23.5) * 3600)
            for _ in range(rng.integers(3, 7)):
                friend = _user_id(rng.integers(0, n_users))
                b.add(t, uid, friend, rng.uniform(800, 6_000), "P2P", device)
                t += rng.integers(30, 10 * 60)

        if pays_rent[u]:                                    # first-time big P2P
            landlord = _user_id(rng.integers(0, n_users))
            rent = rng.uniform(8_000, 40_000)
            for month_start in range(0, cfg.days, 30):
                t = (month_start + rng.integers(0, 5)) * DAY + rng.integers(9, 21) * 3600
                b.add(t, uid, landlord, rent, "P2P", device)

    # ---------------- mule accounts: quiet, aged, normal-looking ----------
    mule_campaign_start = rng.integers(0, cfg.days - 7, cfg.n_mule_accounts)
    for k, m in enumerate(mule_idx):
        uid = all_user_ids[m]
        n_tx = rng.poisson(0.4 * cfg.days)
        for t in rng.integers(0, horizon, n_tx):
            payee = merchants[rng.choice(n_merch, p=merch_pop)]
            b.add(t, uid, payee, np.exp(rng.normal(6.5, 0.8)), "P2M", f"D{m:05d}a")
        # ...and they receive ordinary payments from friends, so their inflow
        # history does not give them away.
        for t in rng.integers(0, horizon, rng.poisson(0.25 * cfg.days)):
            sender = rng.integers(0, n_users)
            b.add(t, _user_id(sender), uid, np.exp(rng.normal(6.8, 0.9)), "P2P", f"D{sender:05d}a")

    def pick_mule(t: int) -> str:
        """Prefer mules whose 7-day campaign window covers time t."""
        day = t // DAY
        active = mule_idx[(mule_campaign_start <= day) & (day < mule_campaign_start + 7)]
        pool = active if len(active) else mule_idx
        return all_user_ids[rng.choice(pool)]

    # ---------------- fraud pattern 1: account takeover / SIM swap --------
    n_ato = int(cfg.ato_incidents_per_1k * n_users / 1000)
    for victim in rng.choice(n_users, n_ato, replace=False):
        uid = _user_id(victim)
        start = rng.integers(DAY, horizon - DAY)
        if rng.random() < 0.5:                              # half happen at night
            start = (start // DAY) * DAY + rng.integers(0, 6 * 3600)
        # 35%: remote-access-app scam -- fraudster drives the victim's OWN phone
        remote_access = rng.random() < 0.35
        fraud_dev = f"D{victim:05d}a" if remote_access else f"F{rng.integers(0, 10**6):06d}"
        mules = [pick_mule(start) for _ in range(rng.integers(1, 3))]
        burst = rng.integers(2, 7)
        t = start
        stealthy = rng.random() < 0.3                       # small amounts to evade rules
        for _ in range(burst):
            amt = rng.uniform(1_500, 6_000) if stealthy else rng.uniform(5_000, 50_000)
            b.add(t, uid, rng.choice(mules), amt, "P2P", fraud_dev, "account_takeover")
            t += rng.integers(60, 15 * 60)

    # ---------------- fraud pattern 2: social-engineering scam ------------
    n_scam = int(cfg.scam_incidents_per_1k * n_users / 1000)
    for victim in rng.choice(n_users, n_scam, replace=False):
        uid = _user_id(victim)
        dev = f"D{victim:05d}a"                             # victim's OWN phone
        day = rng.integers(1, cfg.days)
        t = day * DAY + int(np.clip(rng.normal(14, 3), 8, 22) * 3600)
        mule = pick_mule(t)
        for _ in range(rng.integers(1, 4)):
            # many scams ask for "small" sums (fake fees, refunds) to look normal
            mult = rng.uniform(1.5, 5) if rng.random() < 0.4 else rng.uniform(5, 40)
            amt = min(typical_amt[victim] * mult, 1_00_000)
            kind = "COLLECT" if rng.random() < 0.5 else "P2P"
            b.add(t, uid, mule, max(amt, 499), kind, dev, "scam")
            t += rng.integers(5 * 60, 3 * 3600)

    df = pd.DataFrame(b.data)
    df = df[df["ts"] < horizon].sort_values("ts", kind="stable").reset_index(drop=True)
    df["timestamp"] = pd.Timestamp(cfg.start_date) + pd.to_timedelta(df["ts"], unit="s")
    df.insert(0, "txn_id", [f"T{i:08d}" for i in range(len(df))])
    return df.drop(columns="ts")
