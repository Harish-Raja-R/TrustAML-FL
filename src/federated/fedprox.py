import torch

def fedprox_aggregate(client_updates):
    """
    Same as FedAvg for the server side. The proximal term is added locally during training.
    """
    total_samples = sum(num_samples for _, num_samples in client_updates)
    
    agg_state_dict = {}
    for key in client_updates[0][0].keys():
        agg_state_dict[key] = torch.zeros_like(client_updates[0][0][key], dtype=torch.float)
        
    for state_dict, num_samples in client_updates:
        weight = num_samples / total_samples
        for key in state_dict.keys():
            agg_state_dict[key] += state_dict[key].float() * weight
            
    for key in agg_state_dict.keys():
        agg_state_dict[key] = agg_state_dict[key].to(client_updates[0][0][key].dtype)
        
    return agg_state_dict
