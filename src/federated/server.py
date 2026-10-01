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
        history = {}
        for r in range(1, self.rounds + 1):
            print(f"--- Round {r} ---")
            
            client_updates = []
            
            # Broadcast and Train
            for client in self.clients:
                client.set_parameters(self.global_model.state_dict())
                params, num_samples = client.train()
                client_updates.append((params, num_samples))
                
            # Aggregate
            aggregated_params = self.aggregate(client_updates)
            self.global_model.load_state_dict(aggregated_params)
            
            # Evaluate global model on all clients
            all_y_true = []
            all_preds = []
            for client in self.clients:
                client.set_parameters(self.global_model.state_dict())
                y_true, preds = client.evaluate()
                all_y_true.extend(y_true)
                all_preds.extend(preds)
                
            metrics = evaluate_predictions(np.array(all_y_true), np.array(all_preds))
            print(f"Round {r} Evaluation: PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
            history[r] = metrics
            
        return history
