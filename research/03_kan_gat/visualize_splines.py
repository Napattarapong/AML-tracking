"""
visualize_splines.py
====================
Visualizes the learned KAN spline functions for interpretability.
This is a KEY FIGURE for the paper — it shows exactly what decision
boundaries the KAN learned.

Produces:
  - figures/spline_plot.png: Grid of all KAN spline functions
  - figures/feature_importance.png: Bar chart of feature importance

Usage:
    python visualize_splines.py
"""

import os
import sys
import json
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "03_kan_gat"))

from models import KANGAT, SimpleKANLayer

RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
DATA_PATH = os.path.join(RESULTS_DIR, "aml_graph.pt")
MODEL_PATH = os.path.join(RESULTS_DIR, "kan_gat_best_model.pt")
FEATURE_NAMES_PATH = os.path.join(RESULTS_DIR, "feature_names.json")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_model_and_data():
    """Load the best KAN-GAT model and graph data."""
    data = torch.load(DATA_PATH, weights_only=False)
    
    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_names = json.load(f)
    
    model = KANGAT(
        in_dim=data.x.shape[1],
        gat_hidden_dim=16,
        kan_hidden_dims=[8, 4],
        grid_size=5,
        spline_degree=3,
    )
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
    model.eval()
    
    return model, data, feature_names


def plot_splines(model, data, feature_names, save_path):
    """
    Plot the learned KAN spline functions.

    Shows each input feature's contribution to the risk score
    via the output KAN layer's learned spline.

    Args:
        model: Trained KANGAT model
        data: Graph data object
        feature_names: List of feature names for node features
        save_path: Output plot path
    """
    # Get feature value ranges from data
    with torch.no_grad():
        gat_emb = model.gat_encoder(data.x, data.edge_index)
    gat_dim = gat_emb.shape[1]  # Dynamic: depends on model config
    
    struct_feats = torch.cat(
        [data.out_degree, data.in_degree, data.pagerank, data.clustering_coeff],
        dim=1,
    )
    
    # All feature names: GAT embeddings (gat_dim) + structural (4)
    all_names = [f"GAT_{i}" for i in range(gat_dim)] + [
        "Out-Degree", "In-Degree", "PageRank", "Clustering\nCoefficient"
    ]
    
    # For interpretability, focus on structural features (last 4)
    viz_features = ["Out-Degree", "In-Degree", "PageRank", "Clustering\nCoefficient"]
    viz_indices = [gat_dim, gat_dim + 1, gat_dim + 2, gat_dim + 3]  # Indices in combined input
    
    # Get feature values
    feat_values = struct_feats.numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Use the FIRST KAN layer (kan_layers[0]) for visualization
    # This layer receives the combined [GAT(gat_dim) + struct(4)] = (gat_dim+4)-dim input
    # Indices 0-(gat_dim-1) = GAT embeddings, gat_dim-(gat_dim+3) = structural features
    kan_out = model.kan_layers[0]
    kan_output_layer = model.kan_out  # For output-level importance
    
    for i, (idx, name) in enumerate(zip(viz_indices, viz_features)):
        ax = axes[i]
        
        # Get the range of this feature in the data
        vals = feat_values[:, idx - gat_dim]  # Offset by GAT embedding dim
        x_min, x_max = vals.min(), vals.max()
        x_range = torch.linspace(x_min, x_max, 200)
        x_range_np = x_range.numpy()
        
        # Get spline values
        spline_vals = kan_out.get_spline_function(
            input_idx=idx, output_idx=0, x_range=x_range
        )
        spline_vals = spline_vals.detach().numpy()
        
        # Plot spline
        ax.plot(x_range_np, spline_vals, "b-", linewidth=2.5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        
        # Mark data distribution (rug plot)
        ax.scatter(
            vals, np.full_like(vals, -0.05 * max(abs(spline_vals))),
            c="gray", alpha=0.3, s=10, marker="|"
        )
        
        # Labels
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_xlabel("Standardized Value", fontsize=11)
        ax.set_ylabel("Contribution to\nRisk Score", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add annotation for interpretability
        # Find where spline crosses zero (decision threshold change)
        cross_mask = np.diff(np.sign(spline_vals))
        cross_points = np.where(cross_mask != 0)[0]
        if len(cross_points) > 0 and len(cross_points) < 5:
            for cp in cross_points:
                x_cross = x_range_np[cp]
                ax.axvline(x=x_cross, color="red", linestyle=":", alpha=0.7)
                ax.annotate(
                    f"threshold\n{x_cross:.2f}",
                    xy=(x_cross, 0),
                    xytext=(x_cross + 0.3, 0.2 * max(abs(spline_vals))),
                    fontsize=8,
                    color="red",
                    arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
                )
    
    plt.suptitle(
        "KAN Learned Spline Functions:\nHow Each Structural Feature Contributes to AML Risk",
        fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Spline plot saved to: {save_path}")


def plot_feature_importance(model, data, save_path):
    """
    Compute and plot feature importance from KAN spline ranges.
    
    Importance = range of spline function (max - min) over the data range.
    Larger range = feature has more influence on the output.
    """
    with torch.no_grad():
        gat_emb = model.gat_encoder(data.x, data.edge_index)
    gat_dim = gat_emb.shape[1]
    
    struct_feats = torch.cat(
        [data.out_degree, data.in_degree, data.pagerank, data.clustering_coeff],
        dim=1,
    )
    feat_values = struct_feats.numpy()
    
    viz_indices = [gat_dim, gat_dim + 1, gat_dim + 2, gat_dim + 3]
    viz_names = ["Out-Degree", "In-Degree", "PageRank", "Clustering Coeff"]
    
    # Use first KAN layer (input is (gat_dim+4)-dim: gat_dim GAT + 4 structural)
    kan_out = model.kan_layers[0]
    importances = []
    
    for idx, name in zip(viz_indices, viz_names):
        vals = feat_values[:, idx - gat_dim]
        x_range = torch.linspace(vals.min(), vals.max(), 200)
        spline_vals = kan_out.get_spline_function(
            input_idx=idx, output_idx=0, x_range=x_range
        )
        importance = spline_vals.max() - spline_vals.min()
        importances.append((name, importance.item()))
    
    importances.sort(key=lambda x: x[1], reverse=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [x[0] for x in importances]
    vals = [x[1] for x in importances]
    
    bars = ax.barh(range(len(names)), vals, color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel("Spline Range (max - min)", fontsize=12)
    ax.set_title("KAN Feature Importance\n(Larger Range = More Influence on Risk)", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    
    # Add value labels
    for bar, val in zip(bars, vals):
        ax.text(val + 0.01 * max(vals), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance plot saved to: {save_path}")
    
    return importances


def main():
    print("=" * 60)
    print("KAN Spline Visualization")
    print("=" * 60)
    
    model, data, feature_names = load_model_and_data()
    gat_dim = model.gat_encoder.conv1.in_channels
    print(f"  Model loaded. GAT embedding dim: {gat_dim}, structural features: 4")
    
    plot_splines(model, data, feature_names,
                 os.path.join(FIGURES_DIR, "spline_plot.png"))
    
    importances = plot_feature_importance(model, data,
                 os.path.join(FIGURES_DIR, "feature_importance.png"))
    
    print("\n  Feature Importance Ranking:")
    for name, imp in importances:
        print(f"    {name}: {imp:.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()