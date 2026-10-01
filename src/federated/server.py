import torch
from src.federated.aggregation import fedavg_aggregate
from src.evaluation.metrics import evaluate_predictions
import numpy as np

class FLServer:
    def __init__(self, global_model, clients, rounds=20):
        self.global_model = global_model
        self.clients = clients
        self.rounds = rounds
        
    def aggregate(self, client_updates):
        """
        Default uses FedAvg.
        Can be overridden or parameterized.
        """
        return fedavg_aggregate(client_updates)
        
    def fit(self):
        history = []
        for r in range(1, self.rounds + 1):
            print(f"--- Round {r} ---")
            
            import time
            start_time = time.time()
            client_updates = []
            
            # Broadcast and Train
            for client in self.clients:
                client.set_parameters(self.global_model.state_dict())
                params, num_samples = client.train()
                client_updates.append((params, num_samples))
                
            # Aggregate Deltas
            aggregated_deltas = self.aggregate(client_updates)
            
            # Apply Deltas to global model
            new_state_dict = {}
            for key in self.global_model.state_dict().keys():
                if key in aggregated_deltas and self.global_model.state_dict()[key].dtype in [torch.float16, torch.float32, torch.float64]:
                    new_state_dict[key] = self.global_model.state_dict()[key] + aggregated_deltas[key]
                else:
                    new_state_dict[key] = self.global_model.state_dict()[key]
            
            self.global_model.load_state_dict(new_state_dict)
            
            # Evaluate global model on all clients
            all_y_true = []
            all_preds = []
            for client in self.clients:
                client.set_parameters(self.global_model.state_dict())
                y_true, preds = client.evaluate()
                all_y_true.extend(y_true)
                all_preds.extend(preds)
                
            metrics = evaluate_predictions(np.array(all_y_true), np.array(all_preds))
            latency = (time.time() - start_time) * 1000
            
            # Dummy byte calculation based on param counts (assuming float32 = 4 bytes)
            param_count = sum(p.numel() for p in self.global_model.parameters())
            comm_bytes = param_count * 4 * len(self.clients) * 2 # up and down
            
            metrics["round"] = r
            metrics["round_latency_ms"] = latency
            metrics["communication_bytes"] = comm_bytes
            metrics["participating_clients"] = len(self.clients)
            
            print(f"Round {r} Evaluation: PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
            history.append(metrics)
            
        return history
