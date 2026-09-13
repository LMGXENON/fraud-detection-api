# FraudGuard

Real-time transaction risk scoring service trained on the Kaggle Credit Card Fraud benchmark dataset (`creditcard.csv`). Powered by an unsupervised Isolation Forest model and served via FastAPI with SQLite persistence and a live web dashboard.

---

## Dataset Overview

FraudGuard is trained on the Credit Card Fraud detection benchmark from Kaggle:
- Total Transactions: 284,807 recorded over two days.
- Normal Transactions: 284,315 (99.83%).
- Confirmed Fraud Cases: 492 (0.17%).
- Features: 30 numerical variables (`Time`, `V1` through `V28` PCA components, and `Amount`).
- Ground Truth (`Class`): 0 for legitimate transactions, 1 for confirmed fraud.

---

## Project Structure

```text
fraud-detection-api/
│
├── creditcard.csv        # Kaggle benchmark dataset (284,807 rows, ~144 MB)
├── model.py              # Pure-Python FastIsolationForest anomaly detection algorithm
├── train.py              # Model training pipeline and threshold calibration (~7s)
├── server.py             # FastAPI service with /score, /history, /stats, and Web Dashboard
├── simulator.py          # Streaming simulator replaying Kaggle dataset transactions
├── test_app.py           # Automated end-to-end test suite
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Quickstart Guide

### 1. Install Dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Train the Model
```bash
python3 train.py
```

Expected output:
```text
=================================================================
  FraudGuard - Isolation Forest Training on Kaggle Dataset
=================================================================
Reading 'creditcard.csv' ...
Total rows in Kaggle dataset: 284,807
  - Confirmed Normal transactions : 284,315
  - Confirmed Fraud transactions  : 492

Training on 5000 normal transactions (Unsupervised Baseline)...

Model Evaluation against Real Kaggle Ground Truth:
  Avg Normal Anomaly Score : 0.411 (Baseline Safe)
  Avg Fraud Anomaly Score  : 0.581 (Confirmed Fraud Outliers)
  Separation Ratio         : 1.4x higher anomaly score for fraud

Recommended Flag Threshold : 0.58
Fraud Detection Recall     : 56.3% (277/492 fraud cases flagged)

Trained model saved to 'model.pkl'
=================================================================
```

---

### 3. Run the Service and Web Dashboard

Start the FastAPI application:
```bash
python3 -m uvicorn server:app --port 8000
```

- Web Dashboard: http://127.0.0.1:8000/
- Interactive API Documentation (Swagger UI): http://127.0.0.1:8000/docs
- Alternative API Documentation (ReDoc): http://127.0.0.1:8000/redoc

---

### 4. Run the Payment Feed Simulator

In a second terminal, stream real transactions from `creditcard.csv`:
```bash
python3 simulator.py --delay 0.5 --fraud-boost
```

Console output:
```text
FraudGuard - Live Kaggle Dataset Replay Stream
Target URL: http://127.0.0.1:8000/score (Delay: 0.5s)
==================================================================================
Tx ID   |    Amount | Kaggle Ground Truth   | Model Prediction     | Risk Score
----------------------------------------------------------------------------------
#1      | $  149.62 | [NORMAL PAYMENT]      | APPROVED             | 0.412
#2      | $    2.69 | [NORMAL PAYMENT]      | APPROVED             | 0.395
#3      | $  239.93 | [CONFIRMED FRAUD]     | FLAGGED              | 0.684
#4      | $   12.35 | [NORMAL PAYMENT]      | APPROVED             | 0.401
```

---

## Testing

Run the automated test suite to verify model training, API endpoints, and database logging:
```bash
python3 test_app.py
```

Expected output:
```text
=================================================================
  Running FraudGuard Automated Verification on Kaggle Dataset
=================================================================
[1/4] Training Isolation Forest on 'creditcard.csv' ... PASSED
[2/4] Testing GET / (Service Health) ... PASSED
[3/4] Testing POST /score on Kaggle Normal vs Fraud ... PASSED
[4/4] Testing GET /history & GET /stats (SQLite Persistence) ... PASSED
=================================================================
  ALL TESTS PASSED! Kaggle dataset integration is 100% verified.
=================================================================
```

---

## API Reference

### POST /score
Calculates fraud risk score for an incoming transaction and logs the result to SQLite (`fraudguard.db`).

Request:
```bash
curl -s -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.0, -1.359, -0.072, 2.536, 1.378, -0.338, 0.462, 0.239, 0.098, 0.363, 0.090, -0.551, -0.617, -0.991, -0.311, 1.468, -0.470, 0.207, 0.025, 0.403, 0.251, -0.018, 0.277, -0.110, 0.066, 0.128, -0.189, 0.133, -0.021, 149.62]
  }' | python3 -m json.tool
```

Response:
```json
{
  "transaction_id": 1,
  "amount": 149.62,
  "risk_score": 0.412,
  "flagged": false,
  "status": "APPROVED (NORMAL)"
}
```

---

### GET /history
Retrieves the most recent scored transactions.
```bash
curl -s "http://127.0.0.1:8000/history?limit=5" | python3 -m json.tool
```

---

### GET /stats
Returns aggregate scoring statistics across all recorded transactions.
```bash
curl -s http://127.0.0.1:8000/stats | python3 -m json.tool
```

Response:
```json
{
  "total_scored": 120,
  "flagged_transactions": 22,
  "approved_transactions": 98,
  "flag_rate_percent": 18.33,
  "avg_risk_score": 0.428
}
```

---

## Model Details

- Algorithm: Isolation Forest (unsupervised tree ensemble).
- Objective: Detect anomalies by measuring the number of random binary partitions required to isolate a sample.
- Rationale: Legitimate transactions cluster tightly in feature space and require many splits to isolate. Anomalies and fraudulent transactions diverge from normal patterns and are isolated in significantly fewer splits.
- Threshold: Transactions with an anomaly score of 0.58 or higher are flagged for review.
