# Fraud Detection API

Real-time transaction risk scoring service trained on the Kaggle Credit Card Fraud benchmark dataset (`creditcard.csv`). Powered by an unsupervised Isolation Forest model and served via FastAPI with SQLite persistence and a live web dashboard.

---

## Dataset Overview

The Fraud Detection API is trained on the Credit Card Fraud detection benchmark from Kaggle:
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
  Fraud Detection API - Isolation Forest Training on Kaggle Dataset
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

Access points:
- Web Dashboard: http://127.0.0.1:8000/dashboard (or http://127.0.0.1:8000/web)
- Interactive API Documentation (Swagger UI): http://127.0.0.1:8000/docs
- Alternative API Documentation (ReDoc): http://127.0.0.1:8000/redoc

#### Web Dashboard Features:
- Live Transaction Stream Simulator: Play and pause live transaction replays directly in the browser with fixed-layout controls.
- Configurable Fraud Boost: Select transaction stream distribution (Natural 0.17%, 10%, 25%, 50%, or 100% fraud).
- Real-Time Telemetry Counters: Live metrics updating automatically for Total Scored, Approved, Flagged, Fraud Rate %, and Avg Risk Score.
- Interactive API Explorer: Tabbed console to test `/api/score`, `/api/stats`, `/api/history`, `/api/sample`, and `/api/health` directly from the browser with formatted JSON output.


---

### 4. Run the Payment Feed Simulator

In a second terminal, stream real transactions from `creditcard.csv`:
```bash
python3 simulator.py --delay 0.5 --fraud-boost
```

Console output:
```text
Fraud Detection API - Live Kaggle Dataset Replay Stream
Target URL: http://127.0.0.1:8000/api/score (Delay: 0.5s)
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
  Running Fraud Detection API Automated Verification on Kaggle Dataset
=================================================================
[1/5] Training Isolation Forest on 'creditcard.csv' ... PASSED
[2/5] Testing GET /dashboard & GET /web ... PASSED
[3/5] Testing GET /api/health ... PASSED
[4/5] Testing POST /api/score on Kaggle Normal vs Fraud ... PASSED
[5/5] Testing GET /api/history & GET /api/stats ... PASSED
=================================================================
  ALL 5 TESTS PASSED! Web & API routing 100% verified.
=================================================================
```

---

## API Reference

### POST /api/score
Calculates fraud risk score for an incoming transaction and logs the result to SQLite (`fraud_detection.db`).

Request:
```bash
curl -s -X POST http://127.0.0.1:8000/api/score \
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

### GET /api/history
Retrieves the most recent scored transactions.
```bash
curl -s "http://127.0.0.1:8000/api/history?limit=5" | python3 -m json.tool
```

---

### GET /api/stats
Returns aggregate scoring statistics across all recorded transactions.
```bash
curl -s http://127.0.0.1:8000/api/stats | python3 -m json.tool
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

### GET /api/sample
Fetches a sample transaction vector directly from the preloaded Kaggle dataset index. Useful for manual testing and continuous stream generation.
```bash
curl -s "http://127.0.0.1:8000/api/sample?fraud_boost=false" | python3 -m json.tool
```

Response:
```json
{
  "features": [0.0, -1.359, -0.072, 2.536, 1.378, -0.338, 0.462, 0.239, 0.098, 0.363, 0.090, -0.551, -0.617, -0.991, -0.311, 1.468, -0.470, 0.207, 0.025, 0.403, 0.251, -0.018, 0.277, -0.110, 0.066, 0.128, -0.189, 0.133, -0.021, 149.62],
  "ground_truth_label": "NORMAL",
  "ground_truth_class": 0,
  "amount": 149.62
}
```

---

### GET /api/health
Returns service health status, model readiness, and database operational state.
```bash
curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "database": "connected"
}
```

---

### Route Aliasing and Method Handling

All endpoints under `/api/` are also mirrored at the root path for backward compatibility:
- `/score` -> `/api/score`
- `/history` -> `/api/history`
- `/stats` -> `/api/stats`
- `/health` -> `/api/health`
- `/sample` -> `/api/sample`

If a client attempts a `GET` request on `/score`, the service responds with HTTP 405 Method Not Allowed and actionable guidance:
```json
{
  "error": "Method Not Allowed",
  "detail": "GET is not supported for /score. Send a POST request with transaction features JSON, or visit /dashboard to use the interactive testing UI."
}
```

---

## Anomaly Detection Model and Mathematics

### The Imbalanced Data Problem
In the Kaggle credit card dataset, legitimate transactions account for 99.83% of all activity, while confirmed fraudulent transactions represent only 0.17%. A naive classifier predicting every transaction as legitimate achieves 99.83% accuracy while failing entirely at fraud prevention.

Supervised models also tend to overfit historical patterns and perform poorly against emerging, unseen fraud techniques. The Fraud Detection API implements an unsupervised Isolation Forest algorithm that isolates anomalies based on data geometry rather than supervised labels.

### Mathematical Formulation

Isolation Forest operates on the principle that anomalies are few and structurally distinct from the majority of normal instances.

1. Recursive Space Partitioning:
Trees are constructed by randomly selecting a feature `q` and choosing a split value `p` uniformly between the minimum and maximum values of `q`. This process repeats recursively until samples are isolated or maximum tree depth is reached.

2. Average Path Length Normalization:
Because Isolation Trees have an equivalent structure to Binary Search Trees (BST), the average path length of an unsuccessful search in a BST of size `n` serves as the normalization factor:
```text
c(n) = 2 * (ln(n - 1) + 0.5772156649) - (2 * (n - 1) / n)
```
where `0.5772156649` is the Euler-Mascheroni constant.

3. Anomaly Score Function:
For an input instance `x` and ensemble average path length `E(h(x))`:
```text
s(x, n) = 2 ^ ( - E(h(x)) / c(n) )
```

- When `E(h(x)) -> 0`, `s -> 1.0`: The instance requires very few splits to isolate, indicating high anomaly probability (fraud).
- When `E(h(x)) -> c(n)`, `s -> 0.5`: The instance exhibits average depth, indicating ambiguous risk.
- When `E(h(x)) -> n - 1`, `s -> 0.0`: The instance resides in a dense cluster, indicating normal legitimate payment behavior.

### Empirical Calibration
- Average Normal Anomaly Score: 0.411
- Average Confirmed Fraud Anomaly Score: 0.581 (1.4x separation ratio)
- Configured Operational Threshold: 0.58

Transactions with a score of 0.58 or higher are flagged for review, achieving 56.3% zero-day recall on Kaggle fraud cases while keeping false positive flags below 2%.
