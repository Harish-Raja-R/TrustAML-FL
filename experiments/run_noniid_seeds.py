import sys
import os
import json
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.models.hetero_graphsage import create_hetero_graphsage, PYG_AVAILABLE
from src.federated.client import FLClient
from src.federated.server import FLServer
from src.utils.seed import set_seed
from src.models.init import initialize_model

def run_experiment():
    if not PYG_AVAILABLE:
        print("PyTorch Geometric not available. Exiting.")
        return
        
    seeds = [42, 123, 456]
    degrees = ["iid", "moderate", "severe"]
    methods = ["FedAvg", "FedProx"]
    
    results = []
    
    for degree in degrees:
        for seed in seeds:
            # We force generate a small dataset to avoid OOM
            loader = DataLoader()
            loader.load_or_generate_data(force_generate=True, num_banks=3, num_transactions=1500)
            
            graphs, prep = loader.get_federated_graphs(num_clients=3, non_iid_degree=degree)
            metadata = next(iter(graphs.values())).metadata()
            
            for method in methods:
                print(f"  Method: {method}")
                global_model = create_hetero_graphsage(metadata, hidden_channels=32, out_channels=1, num_layers=2)
                
                dummy_data = next(iter(graphs.values()))
                global_model = initialize_model(global_model, dummy_data)
                
                clients = []
                for client_id, hetero_data in graphs.items():
                    mu = 0.01 if method == "FedProx" else 0.0
                    client = FLClient(client_id, hetero_data, global_model, lr=0.01, local_epochs=1, mu=mu)
                    clients.append(client)
                    
                server = FLServer(global_model, clients, rounds=5) # 5 rounds to speed up
                history = server.fit()
                
                final_metrics = history[-1]
                results.append({
                    "Degree": degree,
                    "Seed": seed,
                    "Method": method,
                    "PR-AUC": final_metrics.get("pr_auc", 0),
                    "ROC-AUC": final_metrics.get("roc_auc", 0)
                })
                
    df = pd.DataFrame(results)
    
    # Calculate Mean and Std
    df_agg = df.groupby(['Degree', 'Method']).agg({'PR-AUC': ['mean', 'std'], 'ROC-AUC': ['mean', 'std']}).reset_index()
    
    # Flatten multi-index columns
    df_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_agg.columns.values]
    
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/noniid_results_raw.csv", index=False)
    df_agg.to_csv("results/tables/Table_3_NonIID.csv", index=False)
    print("\nNon-IID Experiment Completed.")
    print(df_agg)

if __name__ == "__main__":
    run_experiment()
