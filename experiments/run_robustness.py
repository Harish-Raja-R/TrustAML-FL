import sys
import os
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.models.hetero_graphsage import create_hetero_graphsage, PYG_AVAILABLE
from src.federated.client import FLClient
from src.federated.server import FLServer
from src.federated.aggregation import fedavg_aggregate
from src.robustness.median import median_aggregate
from src.robustness.attacks import model_replacement_attack

class MaliciousClient(FLClient):
    def get_parameters(self):
        deltas = super().get_parameters()
        return model_replacement_attack(deltas, target_scale=20.0)

def run_robustness():
    if not PYG_AVAILABLE:
        print("PyTorch Geometric not available. Exiting.")
        return
    from src.models.init import initialize_model
    import torch
    
    print("Loading/Generating data for Robustness experiments...")
    # Generate 10 clients
    loader = DataLoader()
    graphs, prep = loader.get_federated_graphs(num_clients=10, non_iid_degree="iid")
    
    metadata = next(iter(graphs.values())).metadata()
    
    # 0, 10%, 20%, 30% of 10 clients = 0, 1, 2, 3 malicious clients
    malicious_counts = [0, 1, 2, 3]
    aggregators = {
        "FedAvg": fedavg_aggregate,
        "Median": median_aggregate
    }
    
    results = {}
    
    for agg_name, agg_func in aggregators.items():
        for m_count in malicious_counts:
            exp_name = f"{agg_name}_{m_count}0pct_Attacked" if m_count > 0 else f"{agg_name}_Clean"
            # avoid duplicates (e.g. FedAvg_Clean run multiple times)
            if exp_name in results:
                continue
                
            print(f"\n--- Running Robustness Experiment: {exp_name} ---")
            global_model = create_hetero_graphsage(metadata, hidden_channels=32, out_channels=1, num_layers=2)
            
            dummy_data = next(iter(graphs.values()))
            global_model = initialize_model(global_model, dummy_data)
            
            clients = []
            for i, (client_id, hetero_data) in enumerate(graphs.items()):
                if i < m_count:
                    client = MaliciousClient(client_id, hetero_data, global_model, lr=0.01, local_epochs=2)
                else:
                    client = FLClient(client_id, hetero_data, global_model, lr=0.01, local_epochs=2)
                clients.append(client)
                
            server = FLServer(global_model, clients, rounds=10)
            server.aggregate = agg_func
            history = server.fit()
            
            # extract final round PR-AUC and F1 for easy dashboard usage
            final_metrics = history[-1]
            results[exp_name] = {
                "aggregation": agg_name,
                "malicious_percent": m_count * 10,
                "history": history,
                "final_pr_auc": final_metrics.get("pr_auc", 0),
                "final_f1": final_metrics.get("f1", 0)
            }
    
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/robustness.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nRobustness experiment completed. Results saved to results/metrics/robustness.json")

if __name__ == "__main__":
    run_robustness()
