"""
run_xgboost.py
===============
XGBoost baseline: tabular features only, no graph structure.
This establishes whether graph structure adds value — if XGBoost matches
GAT, then we don't need GNNs.

Usage:
    python run_xgboost.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_features():
    """Load tabular features from feature_engineering.py output."""
    path = os.path.join(RESULTS_DIR, "tabular_features.csv")
    if not os.path.exists(path):
        # Run feature engineering if not yet done
        sys.path.insert(0, os.path.join(BASE_DIR, "01_data_prep"))
        from feature_engineering import engineer_features
        df = engineer_features()
    else:
        df = pd.read_csv(path)
    return df


def run_xgboost():
    """Train and evaluate XGBoost classifier."""
    print("=" * 60)
    print("XGBoost Baseline")
    print("=" * 60)

    df = load_features()

    # Feature columns (exclude acct_id and is_sar)
    exclude = {"acct_id", "is_sar"}
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].values.astype(np.float32)
    y = df["is_sar"].values

    print(f"Samples: {len(X)}, Features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    print(f"SAR rate: {y.mean()*100:.1f}%")

    # Handle NaN / inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e8, neginf=-1e8)

    # Stratified 5-fold cross-validation
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    from xgboost import XGBClassifier
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        fbeta_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )

    ap_scores = []
    roc_scores = []
    f2_scores = []
    prec_at_20s = []
    recall_at_20s = []
    all_y_true = []
    all_y_pred_proba = []
    importances = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold + 1}/5")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            use_label_encoder=False,
            verbosity=0,
            random_state=42 + fold,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        ap = average_precision_score(y_test, y_pred_proba)
        roc = roc_auc_score(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        f2 = fbeta_score(y_test, y_pred, beta=2)

        # Precision@20: sort by predicted probability, take top 20
        top20_idx = np.argsort(y_pred_proba)[-20:]
        prec_at_20 = y_test[top20_idx].mean()
        recall_at_20 = y_test[top20_idx].sum() / y_test.sum()

        ap_scores.append(ap)
        roc_scores.append(roc)
        f2_scores.append(f2)
        prec_at_20s.append(prec_at_20)
        recall_at_20s.append(recall_at_20)
        all_y_true.extend(y_test.tolist())
        all_y_pred_proba.extend(y_pred_proba.tolist())
        importances.append(model.feature_importances_)

        print(f"    AUC-PR={ap:.4f}, AUC-ROC={roc:.4f}, F2={f2:.4f}")

    # Aggregate results
    print("\n  Results (5-fold CV):")
    print(f"    AUC-PR:           {np.mean(ap_scores):.4f} ± {np.std(ap_scores):.4f}")
    print(f"    AUC-ROC:          {np.mean(roc_scores):.4f} ± {np.std(roc_scores):.4f}")
    print(f"    F2 Score:         {np.mean(f2_scores):.4f} ± {np.std(f2_scores):.4f}")
    print(f"    Precision@20:     {np.mean(prec_at_20s):.4f} ± {np.std(prec_at_20s):.4f}")
    print(f"    Recall@20:        {np.mean(recall_at_20s):.4f} ± {np.std(recall_at_20s):.4f}")

    # Feature importance
    avg_importance = np.mean(importances, axis=0)
    feat_imp = sorted(
        zip(feature_cols, avg_importance), key=lambda x: x[1], reverse=True
    )
    print("\n  Top-10 Feature Importances:")
    for name, imp in feat_imp[:10]:
        print(f"    {name}: {imp:.4f}")

    # Save results
    results = {
        "model": "XGBoost",
        "auc_pr_mean": float(np.mean(ap_scores)),
        "auc_pr_std": float(np.std(ap_scores)),
        "auc_roc_mean": float(np.mean(roc_scores)),
        "f2_mean": float(np.mean(f2_scores)),
        "precision_at_20_mean": float(np.mean(prec_at_20s)),
        "recall_at_20_mean": float(np.mean(recall_at_20s)),
        "feature_importance": [
            {"name": n, "importance": float(v)} for n, v in feat_imp
        ],
    }
    results_path = os.path.join(RESULTS_DIR, "results_xgboost.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_xgboost()