"""
test_app.py - Automated Verification Suite for Fraud Detection API
-----------------------------------------------------------------
Verifies:
1. Training on 'creditcard.csv' and saving 'model.pkl'.
2. Web Dashboard at GET /dashboard and GET /web.
3. API Health at GET /api/health and sample provider at GET /api/sample.
4. Scoring normal vs fraud transactions at POST /api/score.
5. SQLite persistence at GET /api/history & GET /api/stats.

Run with:
    python3 test_app.py
"""

import os
from fastapi.testclient import TestClient

from train import train, load_kaggle_data, CSV_FILE
from server import app


def run_all_tests():
    print("=" * 65)
    print("  Running Fraud Detection API Automated Verification on Kaggle Dataset")
    print("=" * 65)

    # 1. Test Model Training
    print("[1/5] Training Isolation Forest on 'creditcard.csv' ... ")
    train()
    assert os.path.exists("model.pkl"), "model.pkl was not created!"
    print("      PASSED")

    # Load 1 normal and 1 fraud row from the actual CSV for testing
    normal_sample, fraud_sample = load_kaggle_data(CSV_FILE, max_normal=10)
    normal_tx_features = normal_sample[0]
    fraud_tx_features = fraud_sample[0]

    with TestClient(app) as client:
        # 2. Test Dashboard routes
        print("[2/5] Testing GET /dashboard & GET /web ... ", end="")
        dash_resp = client.get("/dashboard")
        assert dash_resp.status_code == 200
        assert "Fraud Detection API Dashboard" in dash_resp.text

        web_resp = client.get("/web")
        assert web_resp.status_code == 200
        print("PASSED")

        # 3. Test API Health & Sample Endpoint
        print("[3/5] Testing GET /api/health & GET /api/sample ... ", end="")
        health_resp = client.get("/api/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["model_loaded"] is True

        sample_resp = client.get("/api/sample?type=normal")
        assert sample_resp.status_code == 200
        assert len(sample_resp.json()["features"]) == 30
        print("PASSED")

        # 4. Test Scoring Normal vs Fraud via /api/score
        print("[4/5] Testing POST /api/score on Kaggle Normal vs Fraud ... ", end="")
        normal_resp = client.post("/api/score", json={"features": normal_tx_features})
        assert normal_resp.status_code == 200
        n_data = normal_resp.json()

        fraud_resp = client.post("/api/score", json={"features": fraud_tx_features})
        assert fraud_resp.status_code == 200
        f_data = fraud_resp.json()

        assert f_data["risk_score"] > n_data["risk_score"], "Fraud should score higher risk than normal!"
        print(f"PASSED (Normal Risk: {n_data['risk_score']:.2f}, Fraud Risk: {f_data['risk_score']:.2f})")

        # 5. Test Persistence and Stats via /api/...
        print("[5/5] Testing GET /api/history & GET /api/stats (SQLite Persistence) ... ", end="")
        hist_resp = client.get("/api/history?limit=5")
        assert hist_resp.status_code == 200
        assert len(hist_resp.json()) >= 2

        stats_resp = client.get("/api/stats")
        assert stats_resp.status_code == 200
        assert stats_resp.json()["total_scored"] >= 2
        print("PASSED")

    print("=" * 65)
    print("  ALL 5 TESTS PASSED! Live Streamer & API fully verified.")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
