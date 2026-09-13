"""
simulator.py - Real-Time Kaggle Transaction Replay Simulator
------------------------------------------------------------
Streams real transaction rows from 'creditcard.csv' directly to the FraudGuard API.
- Samples both normal transactions and verified fraud rows from the Kaggle dataset.
- Compares Kaggle's ground-truth 'Class' (0 vs 1) with FraudGuard's real-time risk score.
- Prints live color-coded evaluations as transactions stream by.

Usage:
    python3 simulator.py
    python3 simulator.py --delay 0.3     (faster streaming)
    python3 simulator.py --fraud-boost   (oversample fraud rows to see more alerts)
    python3 simulator.py --count 30      (stop after 30 transactions)
"""

import argparse
import csv
import os
import random
import time
import httpx

CSV_FILE = "creditcard.csv"

# ANSI Terminal Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_replay_pool(filename, fraud_boost=False):
    """Loads a pool of transactions from creditcard.csv."""
    if not os.path.exists(filename):
        print(f"{RED}Error: '{filename}' not found!{RESET}")
        exit(1)

    print(f"Loading transactions from '{filename}' ...")
    normal_pool = []
    fraud_pool = []

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Take a representative pool for the simulator
        for row in reader:
            if not row:
                continue
            is_fraud = int(row[-1].strip('"'))
            features = [float(x) for x in row[:-1]]
            amount = float(row[-2])

            if is_fraud == 1:
                fraud_pool.append((features, amount, is_fraud))
            elif len(normal_pool) < 2000:
                normal_pool.append((features, amount, is_fraud))

    print(f"Loaded {len(normal_pool)} normal rows and {len(fraud_pool)} verified fraud rows.")

    if fraud_boost:
        print(f"{YELLOW}[Fraud Boost Enabled]{RESET} Over-sampling fraud rows for live demonstration.")
        # Mix in fraud rows at 25% rate for demo clarity
        combined = normal_pool + (fraud_pool * 10)
    else:
        combined = normal_pool + fraud_pool

    random.shuffle(combined)
    return combined


def main():
    parser = argparse.ArgumentParser(description="FraudGuard Kaggle Stream Simulator")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/score", help="Scoring endpoint")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between transactions")
    parser.add_argument("--count", type=int, default=None, help="Stop after N transactions")
    parser.add_argument("--fraud-boost", action="store_true", help="Inject more fraud rows for demo drama")
    args = parser.parse_args()

    pool = load_replay_pool(CSV_FILE, fraud_boost=args.fraud_boost)

    print(f"\n{CYAN}{BOLD}FraudGuard - Live Kaggle Dataset Replay Stream{RESET}")
    print(f"Target URL: {args.url} (Delay: {args.delay}s)")
    print("=" * 82)
    print(f"{'Tx ID':<7} | {'Amount':>9} | {'Kaggle Ground Truth':<21} | {'Model Prediction':<20} | Risk Score")
    print("-" * 82)

    sent = 0
    timeout_cfg = httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=5.0)

    with httpx.Client(timeout=timeout_cfg) as client:
        try:
            for features, amount, is_fraud in pool:
                payload = {
                    "features": features,
                    "amount": amount
                }

                true_label = f"{RED}[CONFIRMED FRAUD]{RESET}" if is_fraud == 1 else f"{GREEN}[NORMAL PAYMENT]{RESET}"

                try:
                    resp = client.post(args.url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        tx_id = data["transaction_id"]
                        risk = data["risk_score"]
                        flagged = data["flagged"]

                        if flagged:
                            model_pred = f"{RED}{BOLD}🚨 FLAGGED{RESET}"
                        else:
                            model_pred = f"{GREEN}✅ APPROVED{RESET}"

                        print(f"#{tx_id:<6} | ${amount:>8.2f} | {true_label:<30} | {model_pred:<29} | {risk:.3f}")
                    else:
                        print(f"{RED}Server error {resp.status_code}: {resp.text}{RESET}")

                except (httpx.ConnectError, httpx.TimeoutException):
                    print(f"\n{RED}[Connection Error]{RESET} Could not connect to {args.url}")
                    print("Make sure the server is running in another terminal:")
                    print("  python3 -m uvicorn server:app --port 8000\n")
                    break

                sent += 1
                if args.count and sent >= args.count:
                    break

                time.sleep(args.delay)

        except KeyboardInterrupt:
            print(f"\n{CYAN}Replay stopped. Processed {sent} transactions.{RESET}")


if __name__ == "__main__":
    main()
