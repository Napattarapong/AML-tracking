import os
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def get_col(df, possible_names):
    """Helper to dynamically find a column name from a list of possibilities."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def load_aml_graph(base_path):
    print("Loading raw CSV files...")
    
    accounts_path = os.path.join(base_path, 'sample', 'accounts.csv')
    transactions_path = os.path.join(base_path, 'sample', 'transactions.csv')
    alert_accounts_path = os.path.join(base_path, 'sample', 'alert_accounts.csv')
    
    accounts_df = pd.read_csv(accounts_path)
    edges_df = pd.read_csv(transactions_path)
    alerts_df = pd.read_csv(alert_accounts_path)
    
    # Strip any hidden whitespaces and convert to lowercase
    accounts_df.columns = accounts_df.columns.str.strip().str.lower()
    edges_df.columns = edges_df.columns.str.strip().str.lower()
    alerts_df.columns = alerts_df.columns.str.strip().str.lower()
    
    # 1. Dynamically find the exact column names using the new schema
    acct_id_col = get_col(accounts_df, ['acct_id', 'account_id', 'id'])
    alert_id_col = get_col(alerts_df, ['acct_id', 'account_id', 'id'])
    sender_col = get_col(edges_df, ['orig_acct', 'sender_id', 'orig', 'nameorig', 'source'])
    receiver_col = get_col(edges_df, ['bene_acct', 'receiver_id', 'dest', 'namedest', 'target'])
    
    if not acct_id_col:
        raise KeyError(f"Could not find Account ID column. Available: {list(accounts_df.columns)}")
    if not sender_col or not receiver_col:
        raise KeyError(f"Could not find Sender/Receiver columns. Available: {list(edges_df.columns)}")

    print(f"Successfully mapped Node ID -> '{acct_id_col}' | Sender -> '{sender_col}' | Receiver -> '{receiver_col}'")
    print("Preprocessing Nodes and Labels...")
    
    # 2. Create Ground Truth Labels (1 for ML, 0 for Normal)
    if alert_id_col:
        accounts_df['is_ml'] = accounts_df[acct_id_col].isin(alerts_df[alert_id_col]).astype(int)
    else:
        print("Warning: No matching alert column found. Defaulting all labels to 0.")
        accounts_df['is_ml'] = 0
        
    labels = torch.tensor(accounts_df['is_ml'].values, dtype=torch.long)
    
    # 3. Create Continuous Node Indices
    unique_accounts = accounts_df[acct_id_col].unique()
    account_mapping = {acct_id: idx for idx, acct_id in enumerate(unique_accounts)}
    
    # 4. Process Node Features
    feature_cols = []
    bal_col = get_col(accounts_df, ['initial_deposit', 'init_balance', 'balance'])
    type_col = get_col(accounts_df, ['type', 'account_type'])
    country_col = get_col(accounts_df, ['country'])
    
    if bal_col: feature_cols.append(bal_col)
    if type_col: feature_cols.append(type_col)
    if country_col: feature_cols.append(country_col)
        
    node_features_df = pd.get_dummies(accounts_df[feature_cols])
    
    if bal_col and bal_col in node_features_df.columns:
        scaler = StandardScaler()
        node_features_df[bal_col] = scaler.fit_transform(node_features_df[[bal_col]])
    
    # Convert to PyTorch Tensor (Explicitly cast to float32 to fix the numpy.object_ error)
    if not node_features_df.empty:
        x = torch.tensor(node_features_df.astype('float32').values, dtype=torch.float)
    else:
        x = torch.ones((len(unique_accounts), 1), dtype=torch.float)
    
    print("Constructing Edge Index...")
    
    # 5. Map edges to continuous indices
    edges_df['sender_idx'] = edges_df[sender_col].map(account_mapping)
    edges_df['receiver_idx'] = edges_df[receiver_col].map(account_mapping)
    
    # Drop records missing mapping references
    edges_df = edges_df.dropna(subset=['sender_idx', 'receiver_idx'])
    
    source_nodes = torch.tensor(edges_df['sender_idx'].astype('int64').values, dtype=torch.long)
    target_nodes = torch.tensor(edges_df['receiver_idx'].astype('int64').values, dtype=torch.long)
    edge_index = torch.stack([source_nodes, target_nodes], dim=0)
    
    # 6. Extract Edge Features (Amounts)
    amount_col = get_col(edges_df, ['base_amt', 'amount', 'tx_amount', 'tx_amt'])
    if amount_col:
        edge_attr = torch.tensor(edges_df[amount_col].astype('float32').values, dtype=torch.float).view(-1, 1)
    else:
        edge_attr = torch.ones((len(edges_df), 1), dtype=torch.float)
    
    print("Building PyTorch Geometric Data Object...")
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels)
    
    return graph_data

if __name__ == "__main__":
    OUTPUT_DIR = "/Users/blank/Desktop/AMLSim/outputs"
    graph = load_aml_graph(OUTPUT_DIR)
    
    print("\n--- Graph Statistics ---")
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of edges: {graph.num_edges}")
    print(f"Number of node features: {graph.num_node_features}")
    print(f"Number of illicit (ML) accounts: {graph.y.sum().item()}")