import sys
import os
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.models.hetero_graphsage import create_hetero_graphsage, PYG_AVAILABLE
from src.federated.client import FLClient
from src.federated.server import FLServer
from src.privacy.dp import clip_and_add_noise

class DPFLClient(FLClient):
    def __init__(self, client_id, hetero_data, model, lr=0.01, local_epochs=2, clip_norm=1.0, noise_multiplier=0.1):
        super().__init__(client_id, hetero_data, model, lr, local_epochs)
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        
    def get_parameters(self):
        # get_parameters from base class now returns deltas
        deltas = super().get_parameters()
        
        # Clip and add noise to deltas
        dp_deltas = clip_and_add_noise(deltas, self.clip_norm, self.noise_multiplier)
        
        return dp_deltas

def run_privacy():
    if not PYG_AVAILABLE:
        print("PyTorch Geometric not available. Exiting.")
        return
        
    print("Loading/Generating data for DP-FedAvg...")
    loader = DataLoader()
    graphs, prep = loader.get_federated_graphs(num_clients=3, non_iid_degree="moderate")
    
    metadata = next(iter(graphs.values())).metadata()
    
    # We will test Weak DP and Strong DP
    dp_configs = {
        "weak_dp": {"clip_norm": 5.0, "noise_multiplier": 0.01},
        "medium_dp": {"clip_norm": 1.0, "noise_multiplier": 0.1},
        "strong_dp": {"clip_norm": 0.1, "noise_multiplier": 0.5}
    }
    
    results = {}
    
    from src.privacy.accounting import compute_dp_epsilon
    
    for dp_name, config in dp_configs.items():
        print(f"\n--- Running DP Experiment: {dp_name} ---")
        global_model = create_hetero_graphsage(metadata, hidden_channels=32, out_channels=1, num_layers=2)
        
        dummy_data = next(iter(graphs.values()))
        with torch.no_grad():
            global_model(dummy_data.x_dict, dummy_data.edge_index_dict)
            
        clients = []
        for client_id, hetero_data in graphs.items():
            client = DPFLClient(client_id, hetero_data, global_model, lr=0.01, local_epochs=2, 
                                clip_norm=config["clip_norm"], noise_multiplier=config["noise_multiplier"])
            clients.append(client)
            
        server = FLServer(global_model, clients, rounds=15)
        history = server.fit()
        
        # Privacy accounting
        acc = compute_dp_epsilon(
            epochs=15*2, 
            sample_rate=1.0, 
            noise_multiplier=config["noise_multiplier"], 
            delta=1e-5
        )
        
        results[dp_name] = {
            "metrics": history,
            "privacy_status": acc
        }
    
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/privacy.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nPrivacy experiment completed. Results saved to results/metrics/privacy.json")

if __name__ == "__main__":
    run_privacy()
