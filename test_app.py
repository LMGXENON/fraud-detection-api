"""
test_app.py - Automated Verification Suite for FraudGuard (Kaggle Dataset)
--------------------------------------------------------------------------
Verifies:
1. Training on 'creditcard.csv' and saving 'model.pkl'.
2. API health check (GET /).
3. Scoring normal vs fraud transactions from the real Kaggle dataset.
4. SQLite persistence (GET /history, GET /stats).

Run with:
    python3 test_app.py
"""

import os
from fastapi.testclient import TestClient

from train import train, load_kaggle_data, CSV_FILE
from server import app


def run_all_tests():
    print("=" * 65)
    print("  Running FraudGuard Automated Verification on Kaggle Dataset")
    print("=" * 65)

    # 1. Test Model Training
    print("[1/4] Training Isolation Forest on 'creditcard.csv' ... ")
    train()
    assert os.path.exists("model.pkl"), "model.pkl was not created!"
    print("      PASSED")

    # Load 1 normal and 1 fraud row from the actual CSV for testing
    normal_sample, fraud_sample = load_kaggle_data(CSV_FILE, max_normal=10)
    normal_tx_features = normal_sample[0]
    fraud_tx_features = fraud_sample[0]

    with TestClient(app) as client:
        # 2. Test Home endpoint
        print("[2/4] Testing GET / (Service Health) ... ", end="")
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is True
        print("PASSED")

        # 3. Test Scoring Normal vs Fraud
        print("[3/4] Testing POST /score on Kaggle Normal vs Fraud ... ", end="")
        normal_resp = client.post("/score", json={"features": normal_tx_features})
        assert normal_resp.status_code == 200
        n_data = normal_resp.json()

        fraud_resp = client.post("/score", json={"features": fraud_tx_features})
        assert fraud_resp.status_code == 200
        f_data = fraud_resp.json()

        assert f_data["risk_score"] > n_data["risk_score"], "Fraud should score higher risk than normal!"
        print(f"PASSED (Normal Risk: {n_data['risk_score']:.2f}, Fraud Risk: {f_data['risk_score']:.2f})")

        # 4. Test Persistence and Stats
        print("[4/4] Testing GET /history & GET /stats (SQLite Persistence) ... ", end="")
        hist_resp = client.get("/history?limit=5")
        assert hist_resp.status_code == 200
        assert len(hist_resp.json()) >= 2

        stats_resp = client.get("/stats")
        assert stats_resp.status_code == 200
        assert stats_resp.json()["total_scored"] >= 2
        print("PASSED")

    print("=" * 65)
    print("  ALL TESTS PASSED! Kaggle dataset integration is 100% verified.")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
