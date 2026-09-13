"""
server.py - Fraud Detection API & Live Web Dashboard
---------------------------------------------------
Features:
- Live Interactive Web Dashboard: GET /dashboard, GET /web (and redirect from /)
    - Live Stream Simulator Controller: Stable UI (no layout shift), fixed-width buttons,
      and customizable Fraud Boost percentage dropdown (0.17% to 100%).
    - Complete API Explorer: Test all endpoints (/api/score, /api/stats, /api/history,
      /api/health, /api/sample, /docs).
- Resilient Routing:
    Supports both /api/* and root aliases (e.g., /score, /history, /stats, /health).
    Handles GET /score gracefully with instructions instead of 'Method Not Allowed'.
- Isolation Forest anomaly detection model trained on 'creditcard.csv'.
- SQLite persistence ('fraud_detection.db').
"""

import csv
import os
import random
import sqlite3
import pickle
import json
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel, Field
from model import FastIsolationForest

# Serverless & Environment Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IS_SERVERLESS = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

if IS_SERVERLESS:
    DB_FILE = "/tmp/fraud_detection.db"
else:
    DB_FILE = os.environ.get("FRAUD_DB_PATH", os.path.join(BASE_DIR, "fraud_detection.db"))

MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
CSV_FILE = os.path.join(BASE_DIR, "creditcard.csv")
SAMPLE_JSON_FILE = os.path.join(BASE_DIR, "sample_data.json")

model_data = None
sample_normal_pool = []
sample_fraud_pool = []
in_memory_transactions = []


# ---------------------------------------------------------------------------
# Database Layer (with Graceful Serverless Fallback)
# ---------------------------------------------------------------------------
def init_db():
    try:
        db_dir = os.path.dirname(DB_FILE)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
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
    except Exception as e:
        print(f"Notice: SQLite file setup at {DB_FILE}: {e}")


def save_transaction(amount: float, risk_score: float, flagged: bool, details: str) -> int:
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO transactions (timestamp, amount, risk_score, flagged, details)
                VALUES (?, ?, ?, ?, ?)
            """, (
                ts,
                amount,
                risk_score,
                1 if flagged else 0,
                details
            ))
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        tx_id = len(in_memory_transactions) + 1
        in_memory_transactions.append({
            "id": tx_id,
            "timestamp": ts,
            "amount": amount,
            "risk_score": risk_score,
            "flagged": 1 if flagged else 0,
            "details": details
        })
        return tx_id


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
# Resource Loader (Model & Samples)
# ---------------------------------------------------------------------------
def load_resources():
    global model_data, sample_normal_pool, sample_fraud_pool
    init_db()

    # 1. Load trained model from disk if available
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                model_data = pickle.load(f)
            print(f"Kaggle Fraud Model loaded from '{MODEL_FILE}'")
        except Exception as e:
            print(f"Notice: Could not load model from '{MODEL_FILE}': {e}")

    # If model is not loaded, initialize fallback model
    if model_data is None:
        try:
            fallback = FastIsolationForest(n_estimators=30, max_samples=128)
            synthetic = [[0.0] * 29 + [50.0] for _ in range(100)]
            fallback.fit(synthetic)
            model_data = {"model": fallback, "threshold": 0.58, "training_samples": 100}
            print("Initialized self-healing fallback Isolation Forest model.")
        except Exception as e:
            print(f"Warning: Fallback model init failed: {e}")

    # 2. Load sample transactions
    if os.path.exists(CSV_FILE):
        try:
            with open(CSV_FILE, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if not row:
                        continue
                    is_fraud = int(row[-1].strip('"'))
                    feats = [float(x) for x in row[:-1]]
                    amt = float(row[-2])
                    if is_fraud == 1:
                        sample_fraud_pool.append({"features": feats, "amount": amt, "is_fraud": 1})
                    elif len(sample_normal_pool) < 1500:
                        sample_normal_pool.append({"features": feats, "amount": amt, "is_fraud": 0})
            print(f"Web sample pool ready from CSV ({len(sample_normal_pool)} normal, {len(sample_fraud_pool)} fraud)")
        except Exception as e:
            print(f"Notice loading CSV samples: {e}")

    if not sample_normal_pool and os.path.exists(SAMPLE_JSON_FILE):
        try:
            with open(SAMPLE_JSON_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                sample_normal_pool = data.get("normal", [])
                sample_fraud_pool = data.get("fraud", [])
            print(f"Web sample pool ready from JSON ({len(sample_normal_pool)} normal, {len(sample_fraud_pool)} fraud)")
        except Exception as e:
            print(f"Notice loading JSON samples: {e}")


# Initialize immediately for serverless execution
load_resources()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Real-Time Transaction Fraud Scoring API & Web Dashboard",
    version="2.5.0",
    lifespan=lifespan
)


@app.middleware("http")
async def vercel_rewrite_middleware(request: Request, call_next):
    if request.headers.get("x-debug-headers"):
        return JSONResponse({
            "scope_path": request.scope.get("path"),
            "headers": dict(request.headers)
        })
    matched_path = request.headers.get("x-matched-path")
    if matched_path and matched_path not in ("/api/index.py", "/api/index"):
        request.scope["path"] = matched_path
    elif request.scope.get("path") in ("/api/index.py", "/api/index"):
        request.scope["path"] = "/"
    return await call_next(request)



# ---------------------------------------------------------------------------
# Web Dashboard HTML
# ---------------------------------------------------------------------------
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fraud Detection API - Live Dashboard</title>
<style>
  :root {
    --bg: #0b1120;
    --card: #1e293b;
    --card-border: #334155;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --primary: #3b82f6;
    --primary-hover: #2563eb;
    --success: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: var(--bg);
    color: var(--text);
    padding: 24px;
    line-height: 1.5;
  }
  .container { max-width: 1200px; margin: 0 auto; }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--card-border);
    padding-bottom: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
    gap: 12px;
  }
  header h1 { font-size: 24px; font-weight: 700; color: #fff; }
  .badge-live {
    background: rgba(16, 185, 129, 0.15);
    color: var(--success);
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 6px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
  }
  
  /* Stats Cards */
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
    position: relative;
    overflow: hidden;
  }
  .stat-card .label { font-size: 13px; color: var(--text-muted); margin-bottom: 4px; font-weight: 500; }
  .stat-card .value { font-size: 28px; font-weight: 700; }

  /* Streamer Banner - Fixed Layout to Prevent Layout Shifts */
  .streamer-banner {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 18px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    min-height: 84px;
  }
  .streamer-info { flex: 1; min-width: 260px; }
  .streamer-controls {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
  }
  .control-group {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #cbd5e1;
    font-size: 13px;
    white-space: nowrap;
  }
  .btn-stream {
    width: 200px;
    height: 40px;
    font-size: 14px;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.15s ease;
    flex-shrink: 0;
  }
  .btn-stream-start { background: var(--success); color: #fff; }
  .btn-stream-start:hover { background: #059669; }
  .btn-stream-stop { background: var(--danger); color: #fff; }
  .btn-stream-stop:hover { background: #dc2626; }
  
  select, input[type="number"], input[type="text"], textarea {
    background: #0f172a;
    border: 1px solid var(--card-border);
    border-radius: 6px;
    padding: 7px 10px;
    color: #fff;
    font-size: 13px;
    font-family: inherit;
  }
  
  /* Tabs Layout */
  .tabs { display: flex; gap: 8px; margin-bottom: 16px; border-bottom: 1px solid var(--card-border); }
  .tab-btn {
    background: transparent;
    color: var(--text-muted);
    padding: 10px 18px;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    border-top: none; border-left: none; border-right: none;
  }
  .tab-btn.active { color: var(--primary); border-bottom-color: var(--primary); }
  
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  
  .main-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 24px;
  }
  @media (max-width: 900px) {
    .streamer-banner { flex-direction: column; align-items: flex-start; }
    .streamer-controls { width: 100%; flex-wrap: wrap; }
    .btn-stream { width: 100%; }
    .main-grid { grid-template-columns: 1fr; }
  }
  
  .card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 20px;
  }
  .card h2 { font-size: 16px; font-weight: 600; margin-bottom: 16px; color: #fff; }
  
  .btn-group { display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }
  button {
    cursor: pointer;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-size: 12px;
    padding: 7px 12px;
    transition: all 0.15s ease;
  }
  .btn-secondary { background: #334155; color: #fff; }
  .btn-secondary:hover { background: #475569; }
  .btn-danger-light { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
  .btn-danger-light:hover { background: rgba(239, 68, 68, 0.25); }
  .btn-submit { background: var(--primary); color: #fff; width: 100%; padding: 10px; font-size: 14px; margin-top: 10px; }
  .btn-submit:hover { background: var(--primary-hover); }

  label { display: block; font-size: 12px; color: var(--text-muted); margin-bottom: 4px; font-weight: 500; }
  textarea {
    width: 100%;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 11px;
    height: 75px;
    margin-bottom: 10px;
    resize: vertical;
  }
  
  .result-box {
    margin-top: 14px;
    padding: 14px;
    border-radius: 6px;
    background: #0f172a;
    border: 1px solid var(--card-border);
    display: none;
  }
  .result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
  
  /* Tables */
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td { padding: 9px 10px; text-align: left; border-bottom: 1px solid var(--card-border); }
  th { color: var(--text-muted); font-size: 11.5px; font-weight: 600; text-transform: uppercase; }
  .status-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
  }
  .status-pill.approved { background: rgba(16, 185, 129, 0.15); color: var(--success); }
  .status-pill.flagged { background: rgba(239, 68, 68, 0.15); color: var(--danger); }
  
  .json-viewer {
    background: #090e17;
    border: 1px solid var(--card-border);
    border-radius: 6px;
    padding: 14px;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 12px;
    color: #38bdf8;
    max-height: 380px;
    overflow: auto;
    white-space: pre-wrap;
  }
</style>
</head>
<body>

<div class="container">
  <header>
    <div>
      <h1>Fraud Detection API Dashboard</h1>
      <p style="color: var(--text-muted); font-size: 13px;">Real-Time Transaction Risk Scoring & Testing Console</p>
    </div>
    <span class="badge-live">&#9679; API Online (Port 8000)</span>
  </header>

  <!-- Real-Time Metrics Cards -->
  <div class="grid-stats">
    <div class="stat-card">
      <div class="label">Total Scored Transactions</div>
      <div class="value" id="stat-total">--</div>
    </div>
    <div class="stat-card">
      <div class="label">Approved (Normal)</div>
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

  <!-- Live Payment Stream Simulator Controller (Fixed Layout) -->
  <div class="streamer-banner">
    <div class="streamer-info">
      <h3 style="font-size: 16px; margin-bottom: 4px;">Live Transaction Stream Simulator</h3>
      <p style="font-size: 13px; color: var(--text-muted);">
        Stream real transactions from creditcard.csv directly to <code>/api/score</code> and watch numbers count live.
      </p>
    </div>
    <div class="streamer-controls">
      <div class="control-group">
        <span>Speed:</span>
        <select id="streamSpeed">
          <option value="200">0.2s (Fast)</option>
          <option value="500" selected>0.5s (Standard)</option>
          <option value="1000">1.0s (Relaxed)</option>
        </select>
      </div>

      <div class="control-group">
        <span>Fraud Boost:</span>
        <select id="streamFraudRate">
          <option value="0.0017">0.17% (Real Kaggle Rate)</option>
          <option value="0.10">10% Fraud</option>
          <option value="0.25" selected>25% Fraud (Demo Rate)</option>
          <option value="0.50">50% Fraud (Intense)</option>
          <option value="1.00">100% Fraud (Stress Test)</option>
        </select>
      </div>

      <button id="streamToggleBtn" class="btn-stream btn-stream-start" onclick="toggleStream()">Start Continuous Stream</button>
    </div>
  </div>

  <!-- Tabs Navigation -->
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('tab-tester')">Transaction Tester</button>
    <button class="tab-btn" onclick="switchTab('tab-endpoints')">Interactive API Endpoints (All Routes)</button>
  </div>

  <!-- Tab 1: Transaction Tester & Live Feed -->
  <div id="tab-tester" class="tab-content active">
    <div class="main-grid">
      <!-- Tester Form -->
      <div class="card">
        <h2>Manual Transaction Tester (POST /api/score)</h2>
        <div class="btn-group">
          <button type="button" class="btn-secondary" onclick="fetchSample('normal')">Load Real Normal Row</button>
          <button type="button" class="btn-danger-light" onclick="fetchSample('fraud')">Load Real Fraud Row</button>
        </div>

        <form id="scoreForm" onsubmit="submitTransaction(event)">
          <label>Amount (USD):</label>
          <input type="number" step="0.01" id="txAmount" required value="45.00" style="width: 100%; margin-bottom: 10px;">

          <label>30 Kaggle Features [Time, V1..V28, Amount] (JSON Array):</label>
          <textarea id="txFeatures" required></textarea>

          <button type="submit" class="btn-submit">Score Transaction</button>
        </form>

        <div class="result-box" id="resultBox">
          <div class="result-header">
            <span style="font-weight: 700; font-size: 14px;" id="resultStatus">--</span>
            <span style="font-weight: 700; font-size: 18px;" id="resultScore">--</span>
          </div>
          <div style="font-size: 12px; color: var(--text-muted);" id="resultDetails"></div>
        </div>
      </div>

      <!-- Live Recent Feed -->
      <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
          <h2>Live Scored Feed</h2>
          <span style="font-size: 12px; color: var(--text-muted);">Auto-updating every 2s</span>
        </div>
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

  <!-- Tab 2: Interactive API Endpoints (All Routes) -->
  <div id="tab-endpoints" class="tab-content">
    <div class="main-grid">
      <div class="card">
        <h2>Interactive API Endpoint Explorer</h2>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 14px;">
          Click any endpoint to test and inspect the real JSON response.
        </p>
        <div style="display: flex; flex-direction: column; gap: 8px;">
          <button class="btn-secondary" style="padding: 10px; text-align: left;" onclick="callEndpoint('/api/score', 'POST', getSamplePayload())">
            <strong>POST /api/score</strong> (or /score) - Score sample transaction
          </button>
          <button class="btn-secondary" style="padding: 10px; text-align: left;" onclick="callEndpoint('/api/stats', 'GET')">
            <strong>GET /api/stats</strong> (or /stats) - Summary analytics & flag rate
          </button>
          <button class="btn-secondary" style="padding: 10px; text-align: left;" onclick="callEndpoint('/api/history?limit=5', 'GET')">
            <strong>GET /api/history</strong> (or /history) - 5 latest recorded transactions
          </button>
          <button class="btn-secondary" style="padding: 10px; text-align: left;" onclick="callEndpoint('/api/health', 'GET')">
            <strong>GET /api/health</strong> (or /health) - Service & model status
          </button>
          <button class="btn-secondary" style="padding: 10px; text-align: left;" onclick="callEndpoint('/api/sample?type=normal', 'GET')">
            <strong>GET /api/sample?type=normal</strong> - Fetch verified normal vector
          </button>
          <button class="btn-secondary" style="padding: 10px; text-align: left;" onclick="callEndpoint('/api/sample?type=fraud', 'GET')">
            <strong>GET /api/sample?type=fraud</strong> - Fetch verified fraud vector
          </button>
          <a href="/docs" target="_blank" style="text-decoration: none;">
            <button class="btn-secondary" style="padding: 10px; text-align: left; width: 100%; background: #1e3a8a; border: 1px solid #3b82f6;">
              <strong>GET /docs</strong> - Interactive Swagger UI & OpenAPI Specification &rarr;
            </button>
          </a>
        </div>
      </div>

      <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
          <h2>Live Response Output</h2>
          <span style="font-size: 12px; color: var(--text-muted);" id="apiEndpointCalled">None</span>
        </div>
        <div class="json-viewer" id="jsonOutput">Select any endpoint on the left to execute live and view JSON output...</div>
      </div>
    </div>
  </div>
</div>

<script>
let streamInterval = null;
let isStreaming = false;

function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById(tabId).classList.add('active');
}

// Fetch real Kaggle row from API
async function fetchSample(type) {
  try {
    const res = await fetch(`/api/sample?type=${type}`);
    const data = await res.json();
    document.getElementById('txFeatures').value = JSON.stringify(data.features);
    document.getElementById('txAmount').value = data.amount.toFixed(2);
  } catch (err) {
    alert('Could not fetch sample from server.');
  }
}

function getSamplePayload() {
  const rawFeatures = document.getElementById('txFeatures').value;
  const amount = parseFloat(document.getElementById('txAmount').value);
  try {
    return { features: JSON.parse(rawFeatures), amount: amount };
  } catch (e) {
    return { features: [0.0], amount: 45.0 };
  }
}

// Initial sample load
fetchSample('normal');

async function updateStats() {
  try {
    const res = await fetch('/api/stats');
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
    const res = await fetch('/api/history?limit=8');
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
    const res = await fetch('/api/score', {
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

// Continuous Streaming with Percentage Dropdown
function toggleStream() {
  const btn = document.getElementById('streamToggleBtn');
  if (isStreaming) {
    clearInterval(streamInterval);
    isStreaming = false;
    btn.className = 'btn-stream btn-stream-start';
    btn.innerText = 'Start Continuous Stream';
  } else {
    isStreaming = true;
    btn.className = 'btn-stream btn-stream-stop';
    btn.innerText = 'Stop Stream';
    
    const delay = parseInt(document.getElementById('streamSpeed').value);
    streamInterval = setInterval(async () => {
      const fraudProbability = parseFloat(document.getElementById('streamFraudRate').value);
      const isFraud = Math.random() < fraudProbability;
      const type = isFraud ? 'fraud' : 'normal';
      
      try {
        const sampleRes = await fetch(`/api/sample?type=${type}`);
        const sample = await sampleRes.json();
        
        await fetch('/api/score', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ features: sample.features, amount: sample.amount })
        });
        
        updateStats();
        updateHistory();
      } catch (err) {}
    }, delay);
  }
}

async function callEndpoint(url, method, body=null) {
  document.getElementById('apiEndpointCalled').innerText = `${method} ${url}`;
  const out = document.getElementById('jsonOutput');
  out.innerText = 'Executing request...';
  try {
    const opts = { method: method };
    if (body) {
      opts.headers = { 'Content-Type': 'application/json' };
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(url, opts);
    const data = await res.json();
    out.innerText = JSON.stringify(data, null, 2);
  } catch (err) {
    out.innerText = 'Error calling endpoint: ' + err.message;
  }
}

// Initial polling
updateStats();
updateHistory();
setInterval(() => {
  if (!isStreaming) {
    updateStats();
    updateHistory();
  }
}, 2000);
</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Web Dashboard Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
@app.get("/api/index.py", response_class=HTMLResponse, include_in_schema=False)
@app.get("/api/index", response_class=HTMLResponse, include_in_schema=False)
def web_dashboard():
    """Serves the live interactive dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


# ---------------------------------------------------------------------------
# API Endpoints (supports both /api/* and root aliases /*)
# ---------------------------------------------------------------------------
@app.get("/api/health")
@app.get("/api")
@app.get("/health")
def api_health():
    return {
        "service": "Fraud Detection API",
        "dataset": "Kaggle Credit Card Fraud (creditcard.csv)",
        "model_loaded": model_data is not None,
        "endpoints": [
            "POST /api/score (or /score)",
            "GET /api/history (or /history)",
            "GET /api/stats (or /stats)",
            "GET /api/sample"
        ]
    }


@app.get("/api/sample")
def get_sample(type: str = "random"):
    """
    Returns a sample transaction row from creditcard.csv for live testing.
    type: 'normal', 'fraud', or 'random'
    """
    if type == "fraud" and sample_fraud_pool:
        return random.choice(sample_fraud_pool)
    elif type == "normal" and sample_normal_pool:
        return random.choice(sample_normal_pool)
    elif sample_normal_pool and sample_fraud_pool:
        pool = sample_fraud_pool if random.random() < 0.2 else sample_normal_pool
        return random.choice(pool)
    else:
        dummy = [0.0] * 29 + [45.00]
        return {"features": dummy, "amount": 45.00, "is_fraud": 0}


# Handling GET on /score and /api/score to avoid "Method Not Allowed"
@app.get("/score", include_in_schema=False)
@app.get("/api/score", include_in_schema=False)
def get_score_help():
    return JSONResponse(
        status_code=200,
        content={
            "message": "This endpoint requires an HTTP POST request with a JSON transaction payload.",
            "interactive_ui": "Visit /dashboard to score transactions with one click.",
            "post_example": {
                "features": [0.0] * 29 + [45.00],
                "amount": 45.00
            }
        }
    )


@app.post("/api/score", response_model=ScoreResponse)
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


@app.get("/api/history")
@app.get("/history")
def get_history(limit: int = 15):
    try:
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
    except Exception:
        return list(reversed(in_memory_transactions[-limit:]))


@app.get("/api/stats")
@app.get("/stats")
def get_stats():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            total = cursor.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            flagged = cursor.execute("SELECT COUNT(*) FROM transactions WHERE flagged = 1").fetchone()[0]
            avg_risk = cursor.execute("SELECT AVG(risk_score) FROM transactions").fetchone()[0]
    except Exception:
        total = len(in_memory_transactions)
        flagged = sum(1 for t in in_memory_transactions if t.get("flagged") == 1)
        avg_risk = sum(t.get("risk_score", 0.0) for t in in_memory_transactions) / total if total > 0 else 0.0

    flag_rate = round((flagged / total * 100), 2) if total > 0 else 0.0
    return {
        "total_scored": total,
        "flagged_transactions": flagged,
        "approved_transactions": total - flagged,
        "flag_rate_percent": flag_rate,
        "avg_risk_score": round(avg_risk, 3) if avg_risk is not None else 0.0
    }
