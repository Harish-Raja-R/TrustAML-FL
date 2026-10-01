import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.models.classifier import BaselineClassifier
from src.models.mlp import MLP
from src.models.hetero_graphsage import create_hetero_graphsage, PYG_AVAILABLE
from src.evaluation.metrics import evaluate_predictions

def run_classical_baselines(tx_df):
    print("Running Classical Baselines...")
    # Basic features
    features = tx_df[['amount', 'currency', 'transaction_type', 'channel']].values
    labels = tx_df['is_aml'].values
    
    # Simple split (last 20% as test)
    split_idx = int(len(features) * 0.8)
    X_train, X_test = features[:split_idx], features[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]
    
    results = {}
    for model_type in ["lr", "rf"]:
        clf = BaselineClassifier(model_type=model_type)
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)
        metrics = evaluate_predictions(y_test, preds)
        results[model_type] = metrics
        print(f"{model_type.upper()}: PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
        
    return results

def run_gnn_baseline(hetero_data):
    if not PYG_AVAILABLE:
        print("PyTorch Geometric not available. Skipping GNN baseline.")
        return {}
        
    print("Running GNN Baseline (Hetero GraphSAGE)...")
    
    model = create_hetero_graphsage(hetero_data.metadata(), hidden_channels=32, out_channels=1, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train/Test mask for transaction nodes
    num_tx = hetero_data['transaction'].num_nodes
    split_idx = int(num_tx * 0.8)
    
    train_mask = torch.zeros(num_tx, dtype=torch.bool)
    train_mask[:split_idx] = True
    test_mask = torch.zeros(num_tx, dtype=torch.bool)
    test_mask[split_idx:] = True
    
    y = hetero_data['transaction'].y.float()
    
    # Add simple class weight
    pos_weight = (len(y) - y.sum()) / y.sum() if y.sum() > 0 else torch.tensor(1.0)
    
    # Convert edge index to dictionary format required by HeteroData
    edge_index_dict = hetero_data.edge_index_dict
    
    for epoch in range(1, 21):
        model.train()
        optimizer.zero_grad()
        out = model(hetero_data.x_dict, edge_index_dict)
        loss = F.binary_cross_entropy(out['transaction'][train_mask], y[train_mask], weight=torch.where(y[train_mask] == 1, pos_weight, 1.0))
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                out = model(hetero_data.x_dict, edge_index_dict)
                preds = out['transaction'][test_mask].numpy()
                y_true = y[test_mask].numpy()
                metrics = evaluate_predictions(y_true, preds)
                print(f"Epoch {epoch:03d}: Loss: {loss.item():.4f}, PR-AUC: {metrics['pr_auc']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
                
    model.eval()
    with torch.no_grad():
        out = model(hetero_data.x_dict, edge_index_dict)
        preds = out['transaction'][test_mask].numpy()
        y_true = y[test_mask].numpy()
        final_metrics = evaluate_predictions(y_true, preds)
        
    return {"hetero_graphsage": final_metrics}

if __name__ == "__main__":
    os.makedirs("results/metrics", exist_ok=True)
    
    loader = DataLoader()
    print("Loading/Generating data for centralized experiment...")
    tx_df, _ = loader.load_or_generate_data(num_banks=3, num_transactions=5000) # smaller for quick run
    hetero_data, prep = loader.get_centralized_graph()
    
    tx_df = prep.transform(tx_df)
    
    results = {}
    results.update(run_classical_baselines(tx_df))
    results.update(run_gnn_baseline(hetero_data))
    
    with open("results/metrics/centralized.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Centralized experiment completed. Results saved to results/metrics/centralized.json")
