# UPI Fraud Shield - results on unseen test period

Model version `gbm-20261025-201303`. Test period: 60,878 transactions, 113 fraudulent.

## Business impact

| Strategy | Total cost | Fraud prevented | Fraud txns caught | Legit customers challenged | Block rate |
|---|---|---|---|---|---|
| no_fraud_checks | Rs 1,900,014 | 0.0% | 0.0% | 0 | 0.00% |
| rules_only | Rs 904,307 | 52.6% | 34.5% | 111 | 0.05% |
| ml_cost_optimised | Rs 108,436 | 95.6% | 90.3% | 303 | 0.25% |

**ML vs rules:** total cost down Rs 795,871 (88%) over 60,878 payments.

## Detection by fraud pattern (ML)

| Pattern | Fraud txns | Caught (step-up or block) |
|---|---|---|
| account_takeover | 54 | 98.1% |
| scam | 59 | 83.1% |

## Model ranking quality (test)

| Model | PR-AUC | ROC-AUC | Recall @ top 0.5% |
|---|---|---|---|
| logistic_regression | 0.543 | 0.995 | 64.6% |
| gradient_boosting | 0.834 | 0.997 | 88.5% |

## Cost breakdown (ML)

- missed fraud loss: Rs 44,879
- fraud loss after step up: Rs 38,598
- step up friction: Rs 3,630
- wrong block cost: Rs 15,250
- review cost: Rs 6,080

## Drift (PSI, train vs test) - top 5

| Feature | PSI | Status |
|---|---|---|
| user_history_count | 0.166 | watch |
| amount_vs_user_max | 0.116 | watch |
| MODEL_SCORE | 0.016 | stable |
| user_new_payees_24h | 0.008 | stable |
| payee_inflow_per_day | 0.002 | stable |
