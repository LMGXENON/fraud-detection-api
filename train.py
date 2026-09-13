"""
train.py - Fraud Detection API Model Training on Kaggle Credit Card Dataset
--------------------------------------------------------------------------
Trains an Isolation Forest anomaly detector on 'creditcard.csv'.
- Evaluates against the Kaggle ground truth 'Class' label (0 = Normal, 1 = Fraud).
- Saves the trained model to 'model.pkl'.
"""

import csv
import os
import pickle
import random
from model import FastIsolationForest

CSV_FILE = "creditcard.csv"
MODEL_FILE = "model.pkl"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def load_kaggle_data(filename=CSV_FILE, max_normal=5000):
    """
    Loads normal and fraud samples from creditcard.csv.
    Uses 5,000 normal rows for fast, responsive training,
    and loads all confirmed fraud rows for evaluation.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"'{filename}' not found in project directory!")

    print(f"Reading '{filename}' ...")
    normal_rows = []
    fraud_rows = []

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            is_fraud = int(row[-1].strip('"'))
            features = [float(x) for x in row[:-1]]

            if is_fraud == 1:
                fraud_rows.append(features)
            else:
                normal_rows.append(features)

    print(f"Total rows in Kaggle dataset: {len(normal_rows) + len(fraud_rows):,}")
    print(f"  - Confirmed Normal transactions : {len(normal_rows):,}")
    print(f"  - Confirmed Fraud transactions  : {len(fraud_rows):,}")

    training_sample = random.sample(normal_rows, min(len(normal_rows), max_normal))
    return training_sample, fraud_rows


def train():
    print("=" * 65)
    print("  Fraud Detection API - Isolation Forest Training on Kaggle Dataset")
    print("=" * 65)

    X_train, X_fraud = load_kaggle_data(CSV_FILE)
    print(f"\nTraining on {len(X_train)} normal transactions (Unsupervised Baseline)...")

    # Train Isolation Forest
    model = FastIsolationForest(n_estimators=50, max_samples=256)
    model.fit(X_train)

    # Evaluate anomaly scores
    normal_scores = model.decision_function(X_train)
    fraud_scores = model.decision_function(X_fraud)

    avg_normal = sum(normal_scores) / len(normal_scores)
    avg_fraud = sum(fraud_scores) / len(fraud_scores)

    print("\nModel Evaluation against Real Kaggle Ground Truth:")
    print(f"  Avg Normal Anomaly Score : {avg_normal:.3f} (Baseline Safe)")
    print(f"  Avg Fraud Anomaly Score  : {avg_fraud:.3f} (Confirmed Fraud Outliers)")
    print(f"  Separation Ratio         : {avg_fraud / max(0.001, avg_normal):.1f}x higher anomaly score for fraud")

    threshold = 0.58
    detected_fraud = sum(1 for s in fraud_scores if s >= threshold)
    recall = (detected_fraud / len(X_fraud)) * 100
    print(f"\nRecommended Flag Threshold : {threshold}")
    print(f"Fraud Detection Recall     : {recall:.1f}% ({detected_fraud}/{len(X_fraud)} fraud cases flagged)")

    artifact = {
        "model": model,
        "threshold": threshold,
        "feature_count": 30,
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\n✓ Trained model saved to '{MODEL_FILE}'")
    print("=" * 65)


if __name__ == "__main__":
    train()
