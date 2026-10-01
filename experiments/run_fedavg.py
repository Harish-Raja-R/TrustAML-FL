import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.models.hetero_graphsage import create_hetero_graphsage, PYG_AVAILABLE
from src.federated.client import FLClient
from src.federated.server import FLServer

def run_fedavg():
    if not PYG_AVAILABLE:
        print("PyTorch Geometric not available. Exiting.")
        return
        
    print("Loading/Generating data for FedAvg...")
    loader = DataLoader()
    graphs, prep = loader.get_federated_graphs(num_clients=3, non_iid_degree="moderate")
    
    # Initialize global model
    # We use the metadata from the first client's graph
    metadata = next(iter(graphs.values())).metadata()
    global_model = create_hetero_graphsage(metadata, hidden_channels=32, out_channels=1, num_layers=2)
    
    # Initialize clients
    clients = []
    for client_id, hetero_data in graphs.items():
        client = FLClient(client_id, hetero_data, global_model, lr=0.01, local_epochs=2)
        clients.append(client)
        
    server = FLServer(global_model, clients, rounds=20)
    history = server.fit()
    
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/fedavg.json", "w") as f:
        json.dump(history, f, indent=4)
    print("\nFedAvg experiment completed. Results saved to results/metrics/fedavg.json")

if __name__ == "__main__":
    run_fedavg()
