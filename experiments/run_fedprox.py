import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.models.hetero_graphsage import create_hetero_graphsage, PYG_AVAILABLE
from src.federated.client import FLClient
from src.federated.server import FLServer
from src.federated.fedprox import fedprox_aggregate

def run_fedprox():
    if not PYG_AVAILABLE:
        print("PyTorch Geometric not available. Exiting.")
        return
        
    print("Loading/Generating data for FedProx...")
    loader = DataLoader()
    graphs, prep = loader.get_federated_graphs(num_clients=3, non_iid_degree="moderate")
    
    metadata = next(iter(graphs.values())).metadata()
    global_model = create_hetero_graphsage(metadata, hidden_channels=32, out_channels=1, num_layers=2)
    
    import torch
    dummy_data = next(iter(graphs.values()))
    with torch.no_grad():
        global_model(dummy_data.x_dict, dummy_data.edge_index_dict)
    
    clients = []
    for client_id, hetero_data in graphs.items():
        client = FLClient(client_id, hetero_data, global_model, lr=0.01, local_epochs=2, mu=0.1) # mu > 0 activates FedProx
        clients.append(client)
        
    server = FLServer(global_model, clients, rounds=20)
    server.aggregate = fedprox_aggregate
    
    history = server.fit()
    
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/fedprox.json", "w") as f:
        json.dump(history, f, indent=4)
    print("\nFedProx experiment completed. Results saved to results/metrics/fedprox.json")

if __name__ == "__main__":
    run_fedprox()
