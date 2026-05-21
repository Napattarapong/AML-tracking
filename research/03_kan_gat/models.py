"""
models.py
=========
Defines all GNN model architectures for the AML research project:
1. GCNClassifier: Simple GCN with MLP head (baseline)
2. GATMLP: GAT with standard MLP classifier head (baseline)
3. GATLinear: GAT with linear classifier head (ablation)
4. KANGAT: GAT with KAN classification head (OUR METHOD)

All models accept the same interface for drop-in comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

# =========================================================================
# 1. GCN Classifier (Baseline: no attention, standard MLP head)
# =========================================================================


class GCNClassifier(nn.Module):
    """Simple 2-layer GCN with MLP head."""

    def __init__(self, in_dim, hidden_dim=16, out_dim=1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 4, 8),  # +4 for struct features, bottleneck
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, out_dim),
        )

    def forward(self, x, edge_index, data):
        # Graph convolution layers
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index).relu()

        # Concatenate structural features
        struct_feats = torch.cat(
            [
                data.out_degree,
                data.in_degree,
                data.pagerank,
                data.clustering_coeff,
            ],
            dim=1,
        )
        x = torch.cat([x, struct_feats], dim=1)

        # MLP head
        x = self.mlp(x)
        return x


# =========================================================================
# 2. GAT + MLP (Standard GNN: attention + MLP head)
# =========================================================================


class GATEncoder(nn.Module):
    """Shared GAT encoder used by GATMLP, GATLinear, and KANGAT."""

    def __init__(self, in_dim, hidden_dim=16, out_dim=16):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=2, concat=True)
        self.conv2 = GATConv(
            hidden_dim * 2, out_dim, heads=1, concat=False
        )

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GATMLP(nn.Module):
    """GAT encoder + standard MLP classifier head."""

    def __init__(self, in_dim, hidden_dim=16, out_dim=1):
        super().__init__()
        self.encoder = GATEncoder(in_dim, hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 4, 8),  # +4 struct features, bottleneck
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, out_dim),
        )

    def forward(self, x, edge_index, data):
        gat_emb = self.encoder(x, edge_index)
        struct_feats = torch.cat(
            [
                data.out_degree,
                data.in_degree,
                data.pagerank,
                data.clustering_coeff,
            ],
            dim=1,
        )
        combined = torch.cat([gat_emb, struct_feats], dim=1)
        out = self.mlp(combined)
        return out


# =========================================================================
# 3. GAT + Linear (Ablation: no non-linearity in classifier head)
# =========================================================================


class GATLinear(nn.Module):
    """GAT encoder + linear classifier head (ablation)."""

    def __init__(self, in_dim, hidden_dim=16, out_dim=1):
        super().__init__()
        self.encoder = GATEncoder(in_dim, hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim + 4, out_dim)  # +4 struct features

    def forward(self, x, edge_index, data):
        gat_emb = self.encoder(x, edge_index)
        struct_feats = torch.cat(
            [
                data.out_degree,
                data.in_degree,
                data.pagerank,
                data.clustering_coeff,
            ],
            dim=1,
        )
        combined = torch.cat([gat_emb, struct_feats], dim=1)
        out = self.linear(combined)
        return out


# =========================================================================
# 4. KAN-only (Ablation: no graph, only tabular features through KAN)
# =========================================================================


class KANOnly(nn.Module):
    """KAN on tabular features only — no graph structure.

    This uses a learned spline-based classifier (via pykan) on
    raw account features. It tests: do we even need graph structure?
    """

    def __init__(self, feature_dim, num_classes=1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.kan_model = None  # Initialized in train loop via pykan

    def forward(self, x, edge_index=None, data=None):
        # x is expected to be the feature matrix directly
        return x  # Placeholder — actual KAN is trained separately


# =========================================================================
# 5. KAN-GAT (OUR METHOD: GAT + KAN classification head)
# =========================================================================


class SimpleKANLayer(nn.Module):
    """
    A simplified KAN-style layer using B-spline basis functions.

    Implements: out_j = Σ_i φ_{i,j}(x_i)
    where φ is a learned spline function.

    This is a standalone implementation that doesn't depend on pykan internals.
    """

    def __init__(self, in_dim, out_dim, grid_size=5, spline_degree=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_degree = spline_degree

        # Learnable coefficients [out_dim, in_dim, n_bases]
        # n_bases = grid_size + spline_degree (number of B-spline basis functions)
        self.n_bases = grid_size + spline_degree
        self.coeff = nn.Parameter(
            torch.randn(out_dim, in_dim, self.n_bases) * 0.1
        )

        # Base weights (for residual linear connection)
        self.base_w = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.base_b = nn.Parameter(torch.zeros(out_dim))

        # Input normalization to keep values in B-spline range [-2, 2]
        self.input_norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        # Normalize input to keep values in stable range for B-spline grid [-2, 2]
        x_norm = self.input_norm(x)

        # B-spline contribution (on normalized input)
        basis = self._compute_bspline_basis(x_norm)  # [batch, in_dim, n_bases]

        # Apply coefficients to original x for the spline
        # Use x_norm for the B-spline basis, but keep the linear path on original x
        spline_out = torch.einsum("oik,bin->bo", self.coeff, basis)

        # Residual linear connection (on original x)
        linear_out = (
            torch.einsum("oi,bi->bo", self.base_w, x) + self.base_b
        )

        return spline_out + linear_out

    def _compute_bspline_basis(self, x):
        """
        Compute B-spline basis functions using Cox-de Boor recursion.

        Args:
            x: Input tensor [batch_size, in_dim]
        Returns:
            Basis functions [batch_size, in_dim, n_bases]
        """
        B, D = x.shape
        G = self.grid_size
        k = self.spline_degree
        n_bases_total = G + k

        # Fixed uniform grid over [-2, 2] with padding
        h = 4.0 / G
        # Total number of knots needed: n_bases + k + 1 = G + 2k + 1
        n_knots = n_bases_total + k + 1
        grid = torch.linspace(-2 - k*h, 2 + k*h, n_knots, device=x.device)

        x_exp = x.unsqueeze(-1)  # [B, D, 1]

        # Degree 0 basis: [B, D, n_knots - 1]
        bases0 = ((x_exp >= grid[:-1].view(1, 1, -1)) &
                  (x_exp < grid[1:].view(1, 1, -1))).float()

        # Trim to initial n_bases_total
        bases = bases0[..., :n_bases_total]  # [B, D, G+k]

        for d in range(1, k + 1):
            # At degree d, we have n_knots - d - 1 basis functions
            # We only compute up to n_bases_total
            n_current = n_bases_total
            new_bases = torch.zeros_like(bases)

            for i in range(n_current):
                den_left = grid[i + d] - grid[i]
                den_right = grid[i + d + 1] - grid[i + 1]

                left_term = torch.zeros_like(x)
                if den_left > 1e-10:
                    left_term = (x - grid[i]) / den_left * bases[..., i]

                right_term = torch.zeros_like(x)
                if den_right > 1e-10 and (i + 1) < bases.shape[-1]:
                    right_term = (grid[i + d + 1] - x) / den_right * bases[..., i + 1]

                new_bases[..., i] = left_term + right_term

            bases = new_bases

        return bases  # [B, D, G+k]

    def get_spline_function(self, input_idx, output_idx, x_range):
        """
        Get the learned spline function value for a specific input → output pair.
        Used for visualization / interpretability.

        Args:
            input_idx: Which input feature to visualize
            output_idx: Which output neuron
            x_range: Range of input values to evaluate over [n_points]
        Returns:
            Values of the spline function [n_points]
        """
        with torch.no_grad():
            x = x_range.unsqueeze(1)  # [n_points, 1]
            basis = self._compute_bspline_basis(x)  # [n_points, 1, n_bases]
            spline_vals = torch.einsum(
                "oik,pik->po",
                self.coeff[output_idx:output_idx + 1, input_idx:input_idx + 1, :],
                basis,
            ).squeeze()
        linear_vals = (
            x.squeeze() * self.base_w[output_idx, input_idx]
            + self.base_b[output_idx]
        )
        return spline_vals + linear_vals


class KANGAT(nn.Module):
    """
    KAN-GAT: Graph Attention Network with KAN Classification Head.

    Architecture:
        1. GAT encoder produces graph-aware node embeddings
        2. Embeddings + structural features are fed to KAN layers
        3. KAN outputs risk score + interpretable spline functions

    This is the NOVEL contribution of the paper.
    """

    def __init__(
        self,
        in_dim,
        gat_hidden_dim=16,
        kan_hidden_dims=[8, 4],
        num_classes=1,
        grid_size=5,
        spline_degree=3,
    ):
        super().__init__()
        self.num_struct_feats = 4  # out_degree, in_degree, pagerank, clustering

        # GAT Encoder
        self.gat_encoder = GATEncoder(in_dim, gat_hidden_dim, gat_hidden_dim)

        # KAN Fusion layers
        # Input dim = GAT embedding (gat_hidden_dim) + structural features (4)
        kan_input_dim = gat_hidden_dim + self.num_struct_feats

        self.kan_layers = nn.ModuleList()
        prev_dim = kan_input_dim
        for hidden_dim in kan_hidden_dims:
            self.kan_layers.append(
                SimpleKANLayer(prev_dim, hidden_dim, grid_size, spline_degree)
            )
            prev_dim = hidden_dim

        # Output layer
        self.kan_out = SimpleKANLayer(prev_dim, num_classes, grid_size, spline_degree)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, data):
        # 1) GAT encodes graph structure
        gat_emb = self.gat_encoder(x, edge_index)  # [num_nodes, gat_hidden_dim]

        # 2) Concatenate structural features
        struct_feats = torch.cat(
            [
                data.out_degree,
                data.in_degree,
                data.pagerank,
                data.clustering_coeff,
            ],
            dim=1,
        )  # [num_nodes, 4]

        combined = torch.cat([gat_emb, struct_feats], dim=1)  # [num_nodes, gat_hidden_dim + 4]

        # 3) KAN layers
        for kan_layer in self.kan_layers:
            combined = kan_layer(combined)
            combined = F.elu(combined)
            combined = self.dropout(combined)

        out = self.kan_out(combined)
        return out

    def get_spline_functions(self, x_range_dict, layer_idx=-1):
        """
        Get learned spline functions for interpretability.

        Args:
            x_range_dict: dict {feature_name: tensor_of_x_values}
            layer_idx: Which KAN layer to visualize (-1 = output layer)
        Returns:
            dict {feature_name: spline_values}
        """
        kan_layer = self.kan_out if layer_idx == -1 else self.kan_layers[layer_idx]

        # Feature names for the combined input
        feature_names = (
            [f"gat_emb_{i}" for i in range(16)]
            + ["out_degree", "in_degree", "pagerank", "clustering_coeff"]
        )

        results = {}
        for name, x_range in x_range_dict.items():
            if name in feature_names:
                idx = feature_names.index(name)
                spline_vals = kan_layer.get_spline_function(
                    input_idx=idx, output_idx=0, x_range=x_range
                )
                results[name] = spline_vals.detach().numpy()

        return results


# =========================================================================
# Utility: count parameters
# =========================================================================

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(name, in_dim, **kwargs):
    """Factory function for all models."""
    models = {
        "GCN": GCNClassifier,
        "GATMLP": GATMLP,
        "GATLinear": GATLinear,
        "KANGAT": KANGAT,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from {list(models.keys())}")
    return models[name](in_dim=in_dim, **kwargs)