"""Score a few live transactions without starting a server:  python demo.py

Plays three scenarios against the trained system:
  1. a normal grocery payment
  2. an account-takeover burst from a brand-new device at 2 a.m.
  3. a 'refund' collect-request scam on the victim's own phone
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fraudshield.config import DATA_DIR  # noqa: E402
from fraudshield.service import FraudScorer  # noqa: E402


def main() -> None:
    scorer = FraudScorer.load()
    hist = pd.read_csv(DATA_DIR / "transactions.csv.gz", parse_dates=["timestamp"])
    end = hist["timestamp"].max().normalize() + pd.Timedelta(days=1)

    # pick an ordinary customer with a long, clean history
    legit = hist[hist.is_fraud == 0]
    user = legit.user_id.value_counts().index[50]
    mine = legit[legit.user_id == user]
    device = mine.device_id.mode()[0]
    fav_merchant = mine[mine.txn_type == "P2M"].payee_id.mode()[0]
    typical = mine.amount.median()
    # receiving accounts that were used as mules in the historical data
    mules = hist[hist.is_fraud == 1].payee_id.value_counts().index
    print(f"Customer {user}: {len(mine)} past payments, median Rs {typical:,.0f}\n")

    t = lambda h, m=0: (end + pd.Timedelta(hours=h, minutes=m)).isoformat()  # noqa: E731
    scenarios = [
        ("Normal payment to usual shop",
         [dict(timestamp=t(13), payee_id=fav_merchant, amount=round(typical, 2),
               txn_type="P2M", device_id=device)]),
        ("Account takeover: new phone, 2 a.m., rapid transfers",
         [dict(timestamp=t(26, i * 4), payee_id=mules[0], amount=amt, txn_type="P2P",
               device_id="F777777") for i, amt in enumerate([18_000, 24_500, 31_000])]),
        ("Scam: 'refund' collect request on own phone",
         [dict(timestamp=t(39), payee_id=mules[1], amount=round(typical * 15, 2),
               txn_type="COLLECT", device_id=device)]),
    ]
    n = 0
    for title, txns in scenarios:
        print(f"=== {title}")
        for txn in txns:
            n += 1
            res = scorer.score({"txn_id": f"DEMO{n}", "user_id": user, **txn})
            print(f"  Rs {txn['amount']:>9,.0f} -> {res['decision']:<8} "
                  f"risk={res['risk_score']:.4f}  ({res['latency_ms']} ms)")
            for r in res["reasons"]:
                print(f"      - {r['description']}")
        print()
    print("Raw response example:\n" + json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
