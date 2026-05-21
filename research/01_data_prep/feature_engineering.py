"""
feature_engineering.py
======================
Computes additional per-account features from transaction history
for the tabular (XGBoost) baseline and KAN inputs.

Usage:
    python feature_engineering.py
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "outputs", "sample")
SAVE_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(SAVE_DIR, exist_ok=True)


def engineer_features():
    """
    Compute per-account transaction behavior features.
    These capture behavioral patterns not in the graph structure.
    """
    print("=" * 60)
    print("Feature Engineering")
    print("=" * 60)

    # Load data
    accounts = pd.read_csv(os.path.join(OUTPUT_DIR, "accounts.csv"))
    transactions = pd.read_csv(os.path.join(OUTPUT_DIR, "transactions.csv"))
    alerts = pd.read_csv(os.path.join(OUTPUT_DIR, "alert_accounts.csv"))

    accounts.columns = accounts.columns.str.strip().str.lower()
    transactions.columns = transactions.columns.str.strip().str.lower()

    # Parse timestamps
    transactions["step"] = (
        transactions["tran_timestamp"]
        .str.extract(r"(\d+)T")[0]
        .astype(int)
    )

    print(f"[1/4] Computing per-account tx stats...")
    # Group by originating account
    sender_stats = (
        transactions.groupby("orig_acct")
        .agg(
            num_tx=("tran_id", "count"),
            avg_amount=("base_amt", "mean"),
            std_amount=("base_amt", "std"),
            min_amount=("base_amt", "min"),
            max_amount=("base_amt", "max"),
            total_sent=("base_amt", "sum"),
            first_step=("step", "min"),
            last_step=("step", "max"),
            num_unique_recipients=("bene_acct", "nunique"),
            # Ratio of high-value transactions (> 90th percentile of all tx)
        )
        .reset_index()
        .rename(columns={"orig_acct": "acct_id"})
    )

    # Fill NaN std with 0 (accounts with single tx)
    sender_stats["std_amount"] = sender_stats["std_amount"].fillna(0)

    # Compute CV (coefficient of variation) - measure of consistency
    sender_stats["amount_cv"] = sender_stats["std_amount"] / (sender_stats["avg_amount"] + 1e-8)

    # Time span active
    sender_stats["active_steps"] = sender_stats["last_step"] - sender_stats["first_step"] + 1

    # Transactions per step (velocity)
    sender_stats["tx_velocity"] = sender_stats["num_tx"] / (sender_stats["active_steps"] + 1e-8)

    # Recipient diversity
    sender_stats["recipient_diversity"] = (
        sender_stats["num_unique_recipients"] / (sender_stats["num_tx"] + 1e-8)
    )

    print(f"[2/4] Computing per-account receiving stats...")
    # Group by beneficiary account
    receiver_stats = (
        transactions.groupby("bene_acct")
        .agg(
            num_received=("tran_id", "count"),
            total_received=("base_amt", "sum"),
            avg_received=("base_amt", "mean"),
            num_unique_senders=("orig_acct", "nunique"),
        )
        .reset_index()
        .rename(columns={"bene_acct": "acct_id"})
    )

    print(f"[3/4] Merging features...")
    # Merge with accounts
    features = accounts[["acct_id"]].copy()
    features = features.merge(sender_stats, on="acct_id", how="left")
    features = features.merge(receiver_stats, on="acct_id", how="left")

    # Fill NaN (accounts that never sent/received)
    fill_cols = [
        "num_tx", "avg_amount", "std_amount", "min_amount", "max_amount",
        "total_sent", "first_step", "last_step", "num_unique_recipients",
        "amount_cv", "active_steps", "tx_velocity", "recipient_diversity",
        "num_received", "total_received", "avg_received", "num_unique_senders",
    ]
    for col in fill_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    # Alert label
    alerted_ids = set(alerts[alerts["is_sar"] == True]["acct_id"].values)
    features["is_sar"] = features["acct_id"].isin(alerted_ids).astype(int)

    print(f"[4/4] Saving...")
    save_path = os.path.join(SAVE_DIR, "tabular_features.csv")
    features.to_csv(save_path, index=False)
    print(f"  Saved {len(features)} account rows to: {save_path}")
    print(f"  Feature columns: {list(features.columns)}")
    print(f"  SAR rate: {features['is_sar'].mean()*100:.1f}%")
    print("=" * 60)

    return features


if __name__ == "__main__":
    engineer_features()