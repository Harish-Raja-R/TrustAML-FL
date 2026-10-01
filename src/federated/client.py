import torch
import torch.nn.functional as F
import copy

class FLClient:
    def __init__(self, client_id, hetero_data, model, lr=0.01, local_epochs=2, mu=0.0):
        self.client_id = client_id
        self.hetero_data = hetero_data
        self.model = copy.deepcopy(model)
        self.global_model = copy.deepcopy(model)
        self.lr = lr
        self.local_epochs = local_epochs
        self.mu = mu
        
        # Setup split masks
        num_tx = self.hetero_data['transaction'].num_nodes
        split_idx = int(num_tx * 0.8)
        
        self.train_mask = torch.zeros(num_tx, dtype=torch.bool)
        self.train_mask[:split_idx] = True
        self.test_mask = torch.zeros(num_tx, dtype=torch.bool)
        self.test_mask[split_idx:] = True
        
        self.y = self.hetero_data['transaction'].y.float()
        self.pos_weight = (len(self.y) - self.y.sum()) / self.y.sum() if self.y.sum() > 0 else torch.tensor(1.0)
        self.edge_index_dict = self.hetero_data.edge_index_dict
        
    def set_parameters(self, global_params):
        self.model.load_state_dict(copy.deepcopy(global_params))
        self.global_model.load_state_dict(copy.deepcopy(global_params))
        
    def get_parameters(self):
        # Return parameter DELTAS instead of absolute weights
        delta_dict = {}
        for key in self.model.state_dict().keys():
            local_param = self.model.state_dict()[key]
            global_param = self.global_model.state_dict()[key]
            if local_param.dtype in [torch.float16, torch.float32, torch.float64]:
                delta_dict[key] = local_param - global_param
            else:
                delta_dict[key] = local_param.clone()
        return delta_dict
        
    def get_num_samples(self):
        return self.train_mask.sum().item()

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        for _ in range(self.local_epochs):
            optimizer.zero_grad()
            out = self.model(self.hetero_data.x_dict, self.edge_index_dict)
            loss = F.binary_cross_entropy(
                out['transaction'][self.train_mask], 
                self.y[self.train_mask], 
                weight=torch.where(self.y[self.train_mask] == 1, self.pos_weight, 1.0)
            )
            
            # Proximal term
            if self.mu > 0:
                proximal_term = 0.0
                for local_param, global_param in zip(self.model.parameters(), self.global_model.parameters()):
                    proximal_term += ((local_param - global_param) ** 2).sum()
                loss += (self.mu / 2) * proximal_term
                
            loss.backward()
            
            # Optional: DP clipping would happen here
            
            optimizer.step()
            
        return self.get_parameters(), self.get_num_samples()
        
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.hetero_data.x_dict, self.edge_index_dict)
            preds = out['transaction'][self.test_mask].numpy()
            y_true = self.y[self.test_mask].numpy()
            
        return y_true, preds
