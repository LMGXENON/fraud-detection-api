"""
server.py - FraudGuard Real-Time Scoring API (Kaggle Dataset Engine)
-------------------------------------------------------------------
A clean FastAPI service that:
1. Loads the Isolation Forest model trained on 'creditcard.csv'.
2. Scores incoming transactions in real time.
3. Automatically saves every scored transaction to 'fraudguard.db' (SQLite).
4. Provides history and stats endpoints.
"""

import sqlite3
import pickle
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from model import FastIsolationForest

DB_FILE = "fraudguard.db"
MODEL_FILE = "model.pkl"

model_data = None


# ---------------------------------------------------------------------------
# Database Helpers
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
    # Accepts either a 30-item feature array [Time, V1..V28, Amount]
    # OR a dictionary containing 'features' and optional 'amount'
    features: list[float] = Field(..., description="List of 30 transaction features [Time, V1..V28, Amount]")
    amount: Optional[float] = Field(None, description="Transaction dollar amount (if omitted, extracted from features[-1])")


class ScoreResponse(BaseModel):
    transaction_id: int
    amount: float
    risk_score: float
    flagged: bool
    status: str


# ---------------------------------------------------------------------------
# App Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_data
    init_db()
    try:
        with open(MODEL_FILE, "rb") as f:
            model_data = pickle.load(f)
        print(f"✓ Kaggle Fraud Model loaded from '{MODEL_FILE}'")
    except FileNotFoundError:
        print(f"⚠️  '{MODEL_FILE}' not found! Run: python3 train.py")
    yield


app = FastAPI(
    title="FraudGuard API (Kaggle Dataset Edition)",
    description="Scores transactions for fraud risk using an Isolation Forest trained on real credit card data.",
    version="2.1.0",
    lifespan=lifespan
)


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {
        "service": "FraudGuard Real-Time Fraud API",
        "dataset": "Kaggle Credit Card Fraud (creditcard.csv)",
        "model_loaded": model_data is not None,
        "docs_url": "/docs"
    }


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

    # Anomaly score calculation
    raw_anomaly = float(model_data["model"].decision_function([feats])[0])
    risk_score = round(raw_anomaly, 3)
    flagged = risk_score >= model_data["threshold"]

    status = "🚨 FLAGGED FOR FRAUD" if flagged else "✅ APPROVED (NORMAL)"
    details = "Kaggle PCA Anomaly Outlier" if flagged else "Normal Pattern"

    # Persist in database
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
