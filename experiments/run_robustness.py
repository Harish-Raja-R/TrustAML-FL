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
from src.robustness.attacks import gaussian_attack

class MaliciousClient(FLClient):
    def get_parameters(self):
        params = super().get_parameters()
        return gaussian_attack(params, std=5.0)

def run_robustness():
    if not PYG_AVAILABLE:
        print("PyTorch Geometric not available. Exiting.")
        return
        
    print("Loading/Generating data for Robustness experiments...")
    # Generate 5 clients to allow for 1-2 malicious
    loader = DataLoader()
    graphs, prep = loader.get_federated_graphs(num_clients=5, non_iid_degree="iid")
    
    metadata = next(iter(graphs.values())).metadata()
    
    experiments = {
        "FedAvg_Clean": {"aggregation": fedavg_aggregate, "malicious_count": 0},
        "FedAvg_Attacked": {"aggregation": fedavg_aggregate, "malicious_count": 2},
        "Median_Attacked": {"aggregation": median_aggregate, "malicious_count": 2}
    }
    
    results = {}
    
    for exp_name, config in experiments.items():
        print(f"\n--- Running Robustness Experiment: {exp_name} ---")
        global_model = create_hetero_graphsage(metadata, hidden_channels=32, out_channels=1, num_layers=2)
        
        clients = []
        for i, (client_id, hetero_data) in enumerate(graphs.items()):
            if i < config["malicious_count"]:
                client = MaliciousClient(client_id, hetero_data, global_model, lr=0.01, local_epochs=2)
                print(f"  {client_id} is MALICIOUS")
            else:
                client = FLClient(client_id, hetero_data, global_model, lr=0.01, local_epochs=2)
                print(f"  {client_id} is BENIGN")
            clients.append(client)
            
        server = FLServer(global_model, clients, rounds=15)
        server.aggregate = config["aggregation"]
        history = server.fit()
        results[exp_name] = history
    
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/robustness.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nRobustness experiment completed. Results saved to results/metrics/robustness.json")

if __name__ == "__main__":
    run_robustness()
