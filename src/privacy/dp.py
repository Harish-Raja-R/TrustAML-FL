import torch

def clip_and_add_noise(state_dict, clip_norm, noise_multiplier, seed=None):
    """
    Simulates local differential privacy on the client updates.
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    # Calculate global L2 norm of the update
    total_norm = 0.0
    for key, tensor in state_dict.items():
        if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
            total_norm += tensor.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    # Calculate clip factor
    clip_coef = clip_norm / (total_norm + 1e-6)
    clip_coef = min(1.0, clip_coef)
    
    # Clip and add noise
    dp_state_dict = {}
    for key, tensor in state_dict.items():
        if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
            # Clip
            clipped_tensor = tensor * clip_coef
            # Add Gaussian noise
            noise = torch.randn_like(clipped_tensor) * clip_norm * noise_multiplier
            dp_state_dict[key] = clipped_tensor + noise
        else:
            dp_state_dict[key] = tensor.clone()
            
    return dp_state_dict
