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
        # Calculate pseudo-gradient (difference between initial and final weights)
        pseudo_grad = {}
        for key in self.model.state_dict().keys():
            if self.model.state_dict()[key].dtype in [torch.float16, torch.float32, torch.float64]:
                pseudo_grad[key] = self.model.state_dict()[key] - self.global_model.state_dict()[key]
                
        # Clip and add noise to pseudo-gradient
        dp_grad = clip_and_add_noise(pseudo_grad, self.clip_norm, self.noise_multiplier)
        
        # Apply DP gradient back to global weights
        dp_params = {}
        for key in self.model.state_dict().keys():
            if key in dp_grad:
                dp_params[key] = self.global_model.state_dict()[key] + dp_grad[key]
            else:
                dp_params[key] = self.model.state_dict()[key]
                
        return dp_params

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
        results[dp_name] = history
    
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/privacy.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nPrivacy experiment completed. Results saved to results/metrics/privacy.json")

if __name__ == "__main__":
    run_privacy()
