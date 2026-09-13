"""
server.py - FraudGuard Real-Time Scoring API & Live Web Dashboard
-----------------------------------------------------------------
Features:
- Live Web Dashboard at '/' for testing in any browser.
- REST API endpoints: POST /score, GET /history, GET /stats.
- Isolation Forest anomaly detection model trained on 'creditcard.csv'.
- SQLite persistence ('fraudguard.db').
"""

import sqlite3
import pickle
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from model import FastIsolationForest

DB_FILE = "fraudguard.db"
MODEL_FILE = "model.pkl"

model_data = None


# ---------------------------------------------------------------------------
# Database Layer
# ---------------------------------------------------------------------------
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                amount REAL NOT NULL,
                risk_score REAL NOT NULL,
                flagged INTEGER NOT NULL,
                details TEXT
            )
        """)
        conn.commit()


def save_transaction(amount: float, risk_score: float, flagged: bool, details: str) -> int:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transactions (timestamp, amount, risk_score, flagged, details)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            amount,
            risk_score,
            1 if flagged else 0,
            details
        ))
        conn.commit()
        return cursor.lastrowid


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class TransactionPayload(BaseModel):
    features: list[float] = Field(..., description="List of 30 transaction features [Time, V1..V28, Amount]")
    amount: Optional[float] = Field(None, description="Transaction dollar amount (extracted from features[-1] if omitted)")


class ScoreResponse(BaseModel):
    transaction_id: int
    amount: float
    risk_score: float
    flagged: bool
    status: str


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_data
    init_db()
    try:
        with open(MODEL_FILE, "rb") as f:
            model_data = pickle.load(f)
        print(f"Kaggle Fraud Model loaded from '{MODEL_FILE}'")
    except FileNotFoundError:
        print(f"'{MODEL_FILE}' not found. Run: python3 train.py")
    yield


app = FastAPI(
    title="FraudGuard API",
    description="Real-Time Transaction Fraud Scoring API",
    version="2.2.0",
    lifespan=lifespan
)


# ---------------------------------------------------------------------------
# Web Dashboard (Live UI)
# ---------------------------------------------------------------------------
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FraudGuard - Live Fraud Detection Dashboard</title>
<style>
  :root {
    --bg: #0f172a;
    --card: #1e293b;
    --card-border: #334155;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --primary: #3b82f6;
    --primary-hover: #2563eb;
    --success: #10b981;
    --danger: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: var(--bg);
    color: var(--text);
    padding: 24px;
    line-height: 1.5;
  }
  .container { max-width: 1100px; margin: 0 auto; }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--card-border);
    padding-bottom: 16px;
    margin-bottom: 24px;
  }
  header h1 { font-size: 22px; font-weight: 700; color: #fff; }
  .badge-live {
    background: rgba(16, 185, 129, 0.2);
    color: var(--success);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
  }
  .grid-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }
  .stat-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 16px 20px;
  }
  .stat-card .label { font-size: 13px; color: var(--text-muted); margin-bottom: 4px; }
  .stat-card .value { font-size: 24px; font-weight: 700; }
  
  .main-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 24px;
  }
  @media (max-width: 800px) { .main-grid { grid-template-columns: 1fr; } }
  
  .card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 20px;
  }
  .card h2 { font-size: 16px; font-weight: 600; margin-bottom: 16px; color: #fff; }
  
  .btn-group { display: flex; gap: 8px; margin-bottom: 16px; }
  button {
    cursor: pointer;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 14px;
    transition: all 0.15s ease;
  }
  .btn-sample-normal { background: #334155; color: #fff; }
  .btn-sample-normal:hover { background: #475569; }
  .btn-sample-fraud { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
  .btn-sample-fraud:hover { background: rgba(239, 68, 68, 0.3); }
  .btn-submit { background: var(--primary); color: #fff; width: 100%; padding: 10px; font-size: 14px; margin-top: 12px; }
  .btn-submit:hover { background: var(--primary-hover); }

  label { display: block; font-size: 12px; color: var(--text-muted); margin-bottom: 4px; font-weight: 500; }
  input, textarea {
    width: 100%;
    background: #0f172a;
    border: 1px solid var(--card-border);
    border-radius: 6px;
    padding: 8px 12px;
    color: #fff;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 12px;
    margin-bottom: 12px;
  }
  textarea { height: 90px; resize: vertical; }

  /* Result Box */
  .result-box {
    margin-top: 16px;
    padding: 16px;
    border-radius: 6px;
    background: #0f172a;
    border: 1px solid var(--card-border);
    display: none;
  }
  .result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .result-title { font-size: 14px; font-weight: 700; }
  .result-score { font-size: 20px; font-weight: 700; }

  /* History Table */
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--card-border); }
  th { color: var(--text-muted); font-size: 12px; font-weight: 600; text-transform: uppercase; }
  .status-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
  }
  .status-pill.approved { background: rgba(16, 185, 129, 0.2); color: var(--success); }
  .status-pill.flagged { background: rgba(239, 68, 68, 0.2); color: var(--danger); }
</style>
</head>
<body>

<div class="container">
  <header>
    <div>
      <h1>FraudGuard Dashboard</h1>
      <p style="color: var(--text-muted); font-size: 13px;">Real-Time Kaggle Transaction Risk Scoring API</p>
    </div>
    <span class="badge-live">&#9679; API Live (Port 8000)</span>
  </header>

  <!-- Metrics Cards -->
  <div class="grid-stats">
    <div class="stat-card">
      <div class="label">Total Transactions Scored</div>
      <div class="value" id="stat-total">--</div>
    </div>
    <div class="stat-card">
      <div class="label">Approved Transactions</div>
      <div class="value" style="color: var(--success);" id="stat-approved">--</div>
    </div>
    <div class="stat-card">
      <div class="label">Flagged Fraud Anomalies</div>
      <div class="value" style="color: var(--danger);" id="stat-flagged">--</div>
    </div>
    <div class="stat-card">
      <div class="label">Flag Rate</div>
      <div class="value" id="stat-flagrate">--%</div>
    </div>
  </div>

  <div class="main-grid">
    <!-- Test Transaction Card -->
    <div class="card">
      <h2>Interactive Transaction Tester</h2>
      <div class="btn-group">
        <button type="button" class="btn-sample-normal" onclick="loadSample(false)">Load Normal Sample</button>
        <button type="button" class="btn-sample-fraud" onclick="loadSample(true)">Load Fraud Sample</button>
      </div>

      <form id="scoreForm" onsubmit="submitTransaction(event)">
        <label>Amount (USD):</label>
        <input type="number" step="0.01" id="txAmount" required value="45.00">

        <label>30 Kaggle Features [Time, V1..V28, Amount] (JSON Array):</label>
        <textarea id="txFeatures" required></textarea>

        <button type="submit" class="btn-submit">Score Transaction</button>
      </form>

      <div class="result-box" id="resultBox">
        <div class="result-header">
          <span class="result-title" id="resultStatus">--</span>
          <span class="result-score" id="resultScore">--</span>
        </div>
        <div style="font-size: 12px; color: var(--text-muted);" id="resultDetails"></div>
      </div>
    </div>

    <!-- Recent History Card -->
    <div class="card">
      <h2>Recent Scored Transactions</h2>
      <table>
        <thead>
          <tr>
            <th>Tx ID</th>
            <th>Amount</th>
            <th>Risk Score</th>
            <th>Decision</th>
          </tr>
        </thead>
        <tbody id="historyTable">
          <tr><td colspan="4" style="text-align: center; color: var(--text-muted);">Loading transactions...</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<script>
// Sample Kaggle Feature Vectors
const sampleNormal = [0.0, -1.359, -0.072, 2.536, 1.378, -0.338, 0.462, 0.239, 0.098, 0.363, 0.090, -0.551, -0.617, -0.991, -0.311, 1.468, -0.470, 0.207, 0.025, 0.403, 0.251, -0.018, 0.277, -0.110, 0.066, 0.128, -0.189, 0.133, -0.021, 45.00];
const sampleFraud = [406.0, -2.312, 1.951, -1.609, 3.997, -0.522, -1.426, -2.537, 1.391, -2.770, -2.772, 3.202, -2.899, -0.595, -4.289, 0.389, -1.140, -2.830, -0.016, 0.416, 0.126, 0.517, -0.035, -0.465, 0.320, 0.044, 0.177, 0.261, -0.143, 239.93];

function loadSample(isFraud) {
  const sample = isFraud ? sampleFraud : sampleNormal;
  document.getElementById('txFeatures').value = JSON.stringify(sample);
  document.getElementById('txAmount').value = sample[sample.length - 1];
}

// Initial load
loadSample(false);

async function updateStats() {
  try {
    const res = await fetch('/stats');
    if (!res.ok) return;
    const data = await res.json();
    document.getElementById('stat-total').innerText = data.total_scored;
    document.getElementById('stat-approved').innerText = data.approved_transactions;
    document.getElementById('stat-flagged').innerText = data.flagged_transactions;
    document.getElementById('stat-flagrate').innerText = data.flag_rate_percent + '%';
  } catch (e) {}
}

async function updateHistory() {
  try {
    const res = await fetch('/history?limit=8');
    if (!res.ok) return;
    const rows = await res.json();
    const tbody = document.getElementById('historyTable');
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No transactions scored yet.</td></tr>';
      return;
    }
    tbody.innerHTML = rows.map(r => `
      <tr>
        <td>#${r.id}</td>
        <td>$${r.amount.toFixed(2)}</td>
        <td><strong>${r.risk_score.toFixed(3)}</strong></td>
        <td><span class="status-pill ${r.flagged ? 'flagged' : 'approved'}">${r.flagged ? 'FLAGGED' : 'APPROVED'}</span></td>
      </tr>
    `).join('');
  } catch (e) {}
}

async function submitTransaction(e) {
  e.preventDefault();
  const rawFeatures = document.getElementById('txFeatures').value;
  const amount = parseFloat(document.getElementById('txAmount').value);
  let features;
  try {
    features = JSON.parse(rawFeatures);
    if (!Array.isArray(features) || features.length !== 30) throw new Error();
  } catch (err) {
    alert('Error: Features must be a JSON array containing exactly 30 numbers.');
    return;
  }

  try {
    const res = await fetch('/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features, amount })
    });
    const data = await res.json();

    const box = document.getElementById('resultBox');
    const statusEl = document.getElementById('resultStatus');
    const scoreEl = document.getElementById('resultScore');
    const detailsEl = document.getElementById('resultDetails');

    box.style.display = 'block';
    if (data.flagged) {
      box.style.borderColor = 'rgba(239, 68, 68, 0.5)';
      statusEl.style.color = '#ef4444';
      scoreEl.style.color = '#ef4444';
      statusEl.innerText = 'FLAGGED FOR FRAUD';
    } else {
      box.style.borderColor = 'rgba(16, 185, 129, 0.5)';
      statusEl.style.color = '#10b981';
      scoreEl.style.color = '#10b981';
      statusEl.innerText = 'APPROVED (NORMAL)';
    }

    scoreEl.innerText = data.risk_score.toFixed(3);
    detailsEl.innerText = `Tx #${data.transaction_id} | Amount: $${data.amount.toFixed(2)} | Isolation Forest Anomaly Score`;

    updateStats();
    updateHistory();
  } catch (err) {
    alert('Failed to connect to API server.');
  }
}

// Polling interval
updateStats();
updateHistory();
setInterval(() => {
  updateStats();
  updateHistory();
}, 2000);
</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serves the interactive live web dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


@app.post("/score", response_model=ScoreResponse)
def score_transaction(payload: TransactionPayload):
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run: python3 train.py")

    feats = payload.features
    if len(feats) != 30:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 30 features (Time, V1..V28, Amount), got {len(feats)}."
        )

    amount = payload.amount if payload.amount is not None else float(feats[-1])

    # Calculate anomaly score using FastIsolationForest
    raw_anomaly = float(model_data["model"].decision_function([feats])[0])
    risk_score = round(raw_anomaly, 3)
    flagged = risk_score >= model_data["threshold"]

    status = "FLAGGED FOR FRAUD" if flagged else "APPROVED (NORMAL)"
    details = "Kaggle PCA Anomaly Outlier" if flagged else "Normal Pattern"

    # Persist in SQLite
    tx_id = save_transaction(amount, risk_score, flagged, details)

    return ScoreResponse(
        transaction_id=tx_id,
        amount=round(amount, 2),
        risk_score=risk_score,
        flagged=flagged,
        status=status
    )


@app.get("/history")
def get_history(limit: int = 15):
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        rows = cursor.execute("""
            SELECT id, timestamp, amount, risk_score, flagged, details
            FROM transactions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


@app.get("/stats")
def get_stats():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        total = cursor.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        flagged = cursor.execute("SELECT COUNT(*) FROM transactions WHERE flagged = 1").fetchone()[0]
        avg_risk = cursor.execute("SELECT AVG(risk_score) FROM transactions").fetchone()[0]

    flag_rate = round((flagged / total * 100), 2) if total > 0 else 0.0
    return {
        "total_scored": total,
        "flagged_transactions": flagged,
        "approved_transactions": total - flagged,
        "flag_rate_percent": flag_rate,
        "avg_risk_score": round(avg_risk, 3) if avg_risk is not None else 0.0
    }
