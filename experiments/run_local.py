import sys
import os
import json
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.models.classifier import BaselineClassifier
from src.evaluation.metrics import evaluate_predictions
from src.models.hetero_graphsage import create_hetero_graphsage, PYG_AVAILABLE

def run_local_experiments():
    print("Loading/Generating data for local experiments...")
    loader = DataLoader()
    graphs, prep = loader.get_federated_graphs(num_clients=3, non_iid_degree="moderate")
    
    results = {}
    
    for client_id, hetero_data in graphs.items():
        print(f"\n--- Running models for {client_id} ---")
        
        # 1. Classical Baselines
        # We need to extract features from HeteroData or use raw data
        # For simplicity, extract from HeteroData node features
        features = hetero_data['transaction'].x.numpy()
        labels = hetero_data['transaction'].y.numpy()
        
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]
        
        client_res = {}
        for model_type in ["lr", "rf"]:
            clf = BaselineClassifier(model_type=model_type)
            # If train has only one class, skip fitting to avoid errors
            if len(set(y_train)) < 2:
                print(f"Skipping {model_type} for {client_id} due to single class in training.")
                client_res[model_type] = {"pr_auc": 0.0, "roc_auc": 0.0}
                continue
                
            clf.fit(X_train, y_train)
            preds = clf.predict_proba(X_test)
            metrics = evaluate_predictions(y_test, preds)
            client_res[model_type] = metrics
            print(f"{model_type.upper()}: PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
            
        # 2. GNN
        if PYG_AVAILABLE:
            print(f"Running GNN for {client_id}...")
            model = create_hetero_graphsage(hetero_data.metadata(), hidden_channels=32, out_channels=1, num_layers=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            num_tx = hetero_data['transaction'].num_nodes
            train_mask = torch.zeros(num_tx, dtype=torch.bool)
            train_mask[:split_idx] = True
            test_mask = torch.zeros(num_tx, dtype=torch.bool)
            test_mask[split_idx:] = True
            
            y = hetero_data['transaction'].y.float()
            pos_weight = (len(y) - y.sum()) / y.sum() if y.sum() > 0 else torch.tensor(1.0)
            edge_index_dict = hetero_data.edge_index_dict
            
            for epoch in range(1, 21):
                model.train()
                optimizer.zero_grad()
                out = model(hetero_data.x_dict, edge_index_dict)
                loss = F.binary_cross_entropy(out['transaction'][train_mask], y[train_mask], weight=torch.where(y[train_mask] == 1, pos_weight, 1.0))
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                out = model(hetero_data.x_dict, edge_index_dict)
                preds = out['transaction'][test_mask].numpy()
                y_true = y[test_mask].numpy()
                gnn_metrics = evaluate_predictions(y_true, preds)
                print(f"GNN: PR-AUC={gnn_metrics['pr_auc']:.4f}, ROC-AUC={gnn_metrics['roc_auc']:.4f}")
                client_res["hetero_graphsage"] = gnn_metrics
                
        results[client_id] = client_res
        
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/local.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nLocal experiments completed. Results saved to results/metrics/local.json")

if __name__ == "__main__":
    run_local_experiments()
