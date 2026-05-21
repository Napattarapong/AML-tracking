"""
build_graph_dataset.py
=======================
Loads AMLSim outputs, builds PyTorch Geometric graph dataset,
computes node features, edge features, and labels.
Output: res/<torch_geometric.data.Data object for training>

Usage:
    python build_graph_dataset.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import networkx as nx

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "outputs", "sample")
CONF_PATH = os.path.join(os.path.dirname(BASE_DIR), "conf.json")
SAVE_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(SAVE_DIR, exist_ok=True)


def load_data():
    """Load raw CSV files from AMLSim output directory."""
    print("[1/6] Loading raw data...")
    accounts = pd.read_csv(os.path.join(OUTPUT_DIR, "accounts.csv"))
    transactions = pd.read_csv(os.path.join(OUTPUT_DIR, "transactions.csv"))
    alerts = pd.read_csv(os.path.join(OUTPUT_DIR, "alert_accounts.csv"))

    # Clean column names
    accounts.columns = accounts.columns.str.strip().str.lower()
    transactions.columns = transactions.columns.str.strip().str.lower()
    alerts.columns = alerts.columns.str.strip().str.lower()

    print(f"  Accounts: {len(accounts)} rows")
    print(f"  Transactions: {len(transactions)} rows")
    print(f"  Alert accounts: {len(alerts)} rows")
    return accounts, transactions, alerts


def build_labels(accounts, alerts):
    """Create binary labels: 1 = SAR-flagged account, 0 = normal."""
    print("[2/6] Creating labels...")
    alerted_ids = set(alerts[alerts["is_sar"] == True]["acct_id"].values)
    labels = torch.tensor(
        [1 if aid in alerted_ids else 0 for aid in accounts["acct_id"].values],
        dtype=torch.long,
    )
    n_sar = labels.sum().item()
    print(f"  SAR accounts: {n_sar} / {len(labels)} ({n_sar/len(labels)*100:.1f}%)")
    return labels, alerted_ids


def build_node_features(accounts):
    """Build node feature matrix from account attributes."""
    print("[3/6] Building node features...")
    features_list = []

    # 1) Initial deposit (continuous)
    deposit = torch.tensor(
        accounts["initial_deposit"].values, dtype=torch.float
    ).view(-1, 1)
    scaler = StandardScaler()
    deposit_norm = torch.tensor(
        scaler.fit_transform(deposit.numpy()), dtype=torch.float
    )
    features_list.append(deposit_norm)

    # 2) Gender (categorical) — the only non-constant categorical
    gender_dummies = pd.get_dummies(accounts["gender"], prefix="gender")
    gender_tensor = torch.tensor(gender_dummies.values, dtype=torch.float)
    features_list.append(gender_tensor)

    # 3) Age (derived from birth_date)
    accounts["birth_date"] = pd.to_numeric(accounts["birth_date"], errors="coerce")
    accounts["age"] = 2017 - accounts["birth_date"]  # Simulation year is 2017
    age = torch.tensor(accounts["age"].fillna(30).values, dtype=torch.float).view(
        -1, 1
    )
    age_norm = torch.tensor(
        scaler.fit_transform(age.numpy()), dtype=torch.float
    )
    features_list.append(age_norm)

    # NOTE: prior_sar_count is 100% correlated with label (data leakage) — excluded
    # NOTE: type, bank_id, country are all single-value constants — excluded

    # Concatenate all features
    x = torch.cat(features_list, dim=1)
    print(f"  Feature matrix: {x.shape}")
    return x


def build_edge_index(transactions, accounts):
    """Build edge_index from transaction data."""
    print("[4/6] Building edge index...")
    node_id_to_idx = {
        aid: i for i, aid in enumerate(accounts["acct_id"].values)
    }

    src_idx = transactions["orig_acct"].map(node_id_to_idx)
    dst_idx = transactions["bene_acct"].map(node_id_to_idx)

    valid_mask = src_idx.notna() & dst_idx.notna()
    src_idx = src_idx[valid_mask].values.astype(np.int64)
    dst_idx = dst_idx[valid_mask].values.astype(np.int64)

    edge_index = torch.tensor(np.stack([src_idx, dst_idx], axis=0), dtype=torch.long)
    print(f"  Edges: {edge_index.shape[1]}")

    # Edge features: normalized transaction amount
    amounts = transactions.loc[valid_mask, "base_amt"].values.astype(np.float32)
    scaler = StandardScaler()
    amounts_norm = scaler.fit_transform(amounts.reshape(-1, 1))
    edge_attr = torch.tensor(amounts_norm, dtype=torch.float)
    print(f"  Edge attributes: {edge_attr.shape}")

    return edge_index, edge_attr


def compute_structural_features(edge_index, num_nodes):
    """Compute graph structural features for each node."""
    print("[5/6] Computing structural features...")

    # Convert to numpy for degree computation
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    # Out-degree: number of unique recipients
    out_deg = torch.zeros(num_nodes, dtype=torch.float)
    for s in src:
        out_deg[s] += 1.0

    # In-degree: number of unique senders
    in_deg = torch.zeros(num_nodes, dtype=torch.float)
    for d in dst:
        in_deg[d] += 1.0

    # Normalize degrees
    out_deg = (out_deg - out_deg.mean()) / (out_deg.std() + 1e-8)
    in_deg = (in_deg - in_deg.mean()) / (in_deg.std() + 1e-8)

    # Build NetworkX directed graph for advanced features
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for s, d in zip(src, dst):
        G.add_edge(int(s), int(d))

    # PageRank
    print("  Computing PageRank...")
    try:
        pr = nx.pagerank(G, alpha=0.85)
        pagerank = torch.tensor([pr.get(i, 0.0) for i in range(num_nodes)], dtype=torch.float)
    except:
        pagerank = torch.zeros(num_nodes, dtype=torch.float)

    # Clustering coefficient (on undirected version)
    print("  Computing clustering coefficient...")
    G_undirected = G.to_undirected()
    try:
        clustering = nx.clustering(G_undirected)
        clust_coeff = torch.tensor(
            [clustering.get(i, 0.0) for i in range(num_nodes)], dtype=torch.float
        )
    except:
        clust_coeff = torch.zeros(num_nodes, dtype=torch.float)

    struct_features = {
        "out_degree": out_deg.unsqueeze(1),
        "in_degree": in_deg.unsqueeze(1),
        "pagerank": pagerank.unsqueeze(1),
        "clustering_coeff": clust_coeff.unsqueeze(1),
    }
    print(f"  Structural features: {4} dimensions")
    return struct_features


def build_data_object():
    """Main function: build and save the PyG Data object."""
    print("=" * 60)
    print("AMLSim Graph Dataset Builder")
    print("=" * 60)

    accounts, transactions, alerts = load_data()
    labels, alerted_ids = build_labels(accounts, alerts)
    x = build_node_features(accounts)
    edge_index, edge_attr = build_edge_index(transactions, accounts)
    struct_feats = compute_structural_features(edge_index, len(accounts))

    print("[6/6] Creating PyG Data object...")
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels,
        num_nodes=len(accounts),
        out_degree=struct_feats["out_degree"],
        in_degree=struct_feats["in_degree"],
        pagerank=struct_feats["pagerank"],
        clustering_coeff=struct_feats["clustering_coeff"],
        acct_ids=torch.tensor(accounts["acct_id"].values, dtype=torch.long),
        alerted_ids=list(alerted_ids),
    )

    # Save
    save_path = os.path.join(SAVE_DIR, "aml_graph.pt")
    torch.save(data, save_path)
    print(f"\n  Saved to: {save_path}")
    print(f"  Data: {data}")
    print(f"  x shape: {data.x.shape}")
    print(f"  edge_index shape: {data.edge_index.shape}")
    print(f"  edge_attr shape: {data.edge_attr.shape}")
    print(f"  y distribution: {data.y.unique(return_counts=True)}")

    # Save feature names for plotting
    feature_names = _get_feature_names(accounts)
    names_path = os.path.join(SAVE_DIR, "feature_names.json")
    with open(names_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"  Feature names saved to: {names_path}")
    print("=" * 60)

    return data


def _get_feature_names(accounts):
    """Get list of feature names for interpretability plotting."""
    names = ["initial_deposit"]
    for g in sorted(accounts["gender"].unique()):
        names.append(f"gender_{g}")
    names.append("age")
    return names


if __name__ == "__main__":
    build_data_object()