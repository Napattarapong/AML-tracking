"""
run_gcn.py
==========
GCN baseline: simple graph convolutional network with MLP head.
Trains and evaluates with 5-fold cross-validation.

Usage:
    python run_gcn.py
"""

import os
import sys
import json
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    fbeta_score,
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "03_kan_gat"))

from models import GCNClassifier, count_parameters

RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_PATH = os.path.join(RESULTS_DIR, "aml_graph.pt")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameters
HIDDEN_DIM = 16
LR = 0.01
WEIGHT_DECAY = 1e-4
EPOCHS = 200
PATIENCE = 20
SAR_WEIGHT = 20.0
SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_gcn():
    """Train and evaluate GCN with 5-fold CV."""
    print("=" * 60)
    print("GCN Baseline")
    print("=" * 60)

    # Load data
    data = torch.load(DATA_PATH, weights_only=False)
    in_dim = data.x.shape[1]
    num_nodes = data.num_nodes

    print(f"Graph: {num_nodes} nodes, {data.edge_index.shape[1]} edges")
    print(f"Features: {in_dim}-dim")
    print(f"SAR rate: {data.y.float().mean():.3f}")

    # Stratified 5-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    node_indices = np.arange(num_nodes)
    labels = data.y.numpy()

    ap_scores = []
    roc_scores = []
    f2_scores = []
    prec_at_20s = []
    recall_at_20s = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(node_indices, labels)):
        print(f"\n  Fold {fold + 1}/5")
        set_seed(SEED + fold)

        model = GCNClassifier(in_dim=in_dim, hidden_dim=HIDDEN_DIM)
        print(f"    Parameters: {count_parameters(model):,}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        pos_weight = torch.tensor([SAR_WEIGHT])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Training
        best_val_loss = float("inf")
        patience_counter = 0

        # Use train_idx for training, but we need a val split
        # Split train_idx further into train and val
        np.random.shuffle(train_idx)
        val_split = int(0.8 * len(train_idx))
        sub_train_idx = train_idx[:val_split]
        sub_val_idx = train_idx[val_split:]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[sub_train_idx] = True
        val_mask[sub_val_idx] = True
        test_mask[test_idx] = True

        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data)
            loss = criterion(out[train_mask], data.y[train_mask].float().unsqueeze(1))
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index, data)
                val_loss = criterion(
                    val_out[val_mask], data.y[val_mask].float().unsqueeze(1)
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        # Evaluation
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data)
            y_pred = torch.sigmoid(out[test_mask]).squeeze().numpy()
            y_true = data.y[test_mask].numpy()

        ap = average_precision_score(y_true, y_pred)
        roc = roc_auc_score(y_true, y_pred)
        y_pred_bin = (y_pred >= 0.5).astype(int)
        f2 = fbeta_score(y_true, y_pred_bin, beta=2)

        # Precision@20
        top20_idx = np.argsort(y_pred)[-20:]
        prec_at_20 = y_true[top20_idx].mean()
        recall_at_20 = y_true[top20_idx].sum() / max(y_true.sum(), 1)

        ap_scores.append(ap)
        roc_scores.append(roc)
        f2_scores.append(f2)
        prec_at_20s.append(prec_at_20)
        recall_at_20s.append(recall_at_20)

        print(f"    AUC-PR={ap:.4f}, AUC-ROC={roc:.4f}, F2={f2:.4f}")
        print(f"    Precision@20={prec_at_20:.4f}, Recall@20={recall_at_20:.4f}")

    # Results
    print("\n  Results (5-fold CV):")
    print(f"    AUC-PR:           {np.mean(ap_scores):.4f} ± {np.std(ap_scores):.4f}")
    print(f"    AUC-ROC:          {np.mean(roc_scores):.4f} ± {np.std(roc_scores):.4f}")
    print(f"    F2 Score:         {np.mean(f2_scores):.4f} ± {np.std(f2_scores):.4f}")
    print(f"    Precision@20:     {np.mean(prec_at_20s):.4f} ± {np.std(prec_at_20s):.4f}")
    print(f"    Recall@20:        {np.mean(recall_at_20s):.4f} ± {np.std(recall_at_20s):.4f}")

    results = {
        "model": "GCN",
        "auc_pr_mean": float(np.mean(ap_scores)),
        "auc_pr_std": float(np.std(ap_scores)),
        "auc_roc_mean": float(np.mean(roc_scores)),
        "f2_mean": float(np.mean(f2_scores)),
        "precision_at_20_mean": float(np.mean(prec_at_20s)),
        "recall_at_20_mean": float(np.mean(recall_at_20s)),
    }
    results_path = os.path.join(RESULTS_DIR, "results_gcn.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    train_gcn()