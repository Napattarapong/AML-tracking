"""
train_kan_gat.py
================
Trains the KAN-GAT model (OUR METHOD) with 5-fold cross-validation.
KAN-GAT replaces the standard MLP classification head of a GAT with
a Kolmogorov-Arnold Network using learnable spline functions.

Usage:
    python train_kan_gat.py
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
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "03_kan_gat"))

from models import KANGAT, count_parameters

RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
DATA_PATH = os.path.join(RESULTS_DIR, "aml_graph.pt")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Hyperparameters
GAT_HIDDEN = 16
KAN_HIDDEN = [8, 4]  # Two KAN hidden layers
LR = 0.005
WEIGHT_DECAY = 1e-4
EPOCHS = 200
PATIENCE = 20
SAR_WEIGHT = 20.0
GRID_SIZE = 5
SPLINE_DEGREE = 3
SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_kan_gat():
    """Train and evaluate KAN-GAT with 5-fold CV."""
    print("=" * 60)
    print("KAN-GAT (OUR METHOD)")
    print("=" * 60)

    data = torch.load(DATA_PATH, weights_only=False)
    in_dim = data.x.shape[1]
    num_nodes = data.num_nodes

    print(f"Graph: {num_nodes} nodes, {data.edge_index.shape[1]} edges")
    print(f"Features: {in_dim}-dim")
    print(f"SAR rate: {data.y.float().mean():.3f}")
    print(f"KAN grid size: {GRID_SIZE}, spline degree: {SPLINE_DEGREE}")
    print(f"KAN hidden layers: {KAN_HIDDEN}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    node_indices = np.arange(num_nodes)
    labels = data.y.numpy()

    ap_scores = []
    roc_scores = []
    f2_scores = []
    prec_at_20s = []
    recall_at_20s = []
    all_models = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(node_indices, labels)):
        print(f"\n  Fold {fold + 1}/5")
        set_seed(SEED + fold)

        model = KANGAT(
            in_dim=in_dim,
            gat_hidden_dim=GAT_HIDDEN,
            kan_hidden_dims=KAN_HIDDEN,
            grid_size=GRID_SIZE,
            spline_degree=SPLINE_DEGREE,
        )
        print(f"    Parameters: {count_parameters(model):,}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        pos_weight = torch.tensor([SAR_WEIGHT])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss = float("inf")
        patience_counter = 0

        # Train/val split
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index, data)
                val_loss = criterion(
                    val_out[val_mask], data.y[val_mask].float().unsqueeze(1)
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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

        top20_idx = np.argsort(y_pred)[-20:]
        prec_at_20 = y_true[top20_idx].mean()
        recall_at_20 = y_true[top20_idx].sum() / max(y_true.sum(), 1)

        ap_scores.append(ap)
        roc_scores.append(roc)
        f2_scores.append(f2)
        prec_at_20s.append(prec_at_20)
        recall_at_20s.append(recall_at_20)
        all_models.append(model.cpu())

        print(f"    AUC-PR={ap:.4f}, AUC-ROC={roc:.4f}, F2={f2:.4f}")
        print(f"    Precision@20={prec_at_20:.4f}, Recall@20={recall_at_20:.4f}")

    # Results
    print("\n  Results (5-fold CV):")
    print(f"    AUC-PR:           {np.mean(ap_scores):.4f} ± {np.std(ap_scores):.4f}")
    print(f"    AUC-ROC:          {np.mean(roc_scores):.4f} ± {np.std(roc_scores):.4f}")
    print(f"    F2 Score:         {np.mean(f2_scores):.4f} ± {np.std(f2_scores):.4f}")
    print(f"    Precision@20:     {np.mean(prec_at_20s):.4f} ± {np.std(prec_at_20s):.4f}")
    print(f"    Recall@20:        {np.mean(recall_at_20s):.4f} ± {np.std(recall_at_20s):.4f}")

    # Save results
    results = {
        "model": "KAN-GAT",
        "config": {
            "gat_hidden_dim": GAT_HIDDEN,
            "kan_hidden_dims": KAN_HIDDEN,
            "grid_size": GRID_SIZE,
            "spline_degree": SPLINE_DEGREE,
            "sar_weight": SAR_WEIGHT,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
        },
        "auc_pr_mean": float(np.mean(ap_scores)),
        "auc_pr_std": float(np.std(ap_scores)),
        "auc_roc_mean": float(np.mean(roc_scores)),
        "f2_mean": float(np.mean(f2_scores)),
        "precision_at_20_mean": float(np.mean(prec_at_20s)),
        "recall_at_20_mean": float(np.mean(recall_at_20s)),
    }
    results_path = os.path.join(RESULTS_DIR, "results_KAN_GAT.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # Save best model
    best_fold = np.argmax(ap_scores)
    model_path = os.path.join(RESULTS_DIR, "kan_gat_best_model.pt")
    torch.save(all_models[best_fold].state_dict(), model_path)
    print(f"  Best model (fold {best_fold + 1}) saved to: {model_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    train_kan_gat()