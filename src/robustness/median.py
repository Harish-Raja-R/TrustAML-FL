import torch

def median_aggregate(client_updates):
    """
    Coordinate-wise median aggregation.
    client_updates: list of tuples (state_dict, num_samples)
    """
    agg_state_dict = {}
    
    # We ignore num_samples for median (standard median doesn't weight, though weighted median exists)
    for key in client_updates[0][0].keys():
        if client_updates[0][0][key].dtype in [torch.float16, torch.float32, torch.float64]:
            stacked = torch.stack([update[0][key] for update in client_updates])
            agg_state_dict[key] = torch.median(stacked, dim=0).values
        else:
            # For integer tensors (e.g., num_batches_tracked), just take the first one or majority
            agg_state_dict[key] = client_updates[0][0][key].clone()
            
    return agg_state_dict
