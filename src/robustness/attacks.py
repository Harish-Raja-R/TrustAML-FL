import torch

def sign_flip_attack(delta_dict, scale=-1.0):
    """
    Inverts the update deltas of the client.
    delta_k = local_model - global_model
    By inverting this, the client pushes the model away from its local optima.
    """
    attacked_dict = {}
    for key, tensor in delta_dict.items():
        if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
            attacked_dict[key] = tensor * scale
        else:
            attacked_dict[key] = tensor.clone()
    return attacked_dict

def gaussian_attack(delta_dict, std=5.0):
    """
    Adds Gaussian noise to the client's update delta.
    """
    attacked_dict = {}
    for key, tensor in delta_dict.items():
        if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
            attacked_dict[key] = tensor + torch.randn_like(tensor) * std
        else:
            attacked_dict[key] = tensor.clone()
    return attacked_dict

def model_replacement_attack(delta_dict, target_scale=10.0):
    """
    Scales the update dramatically in an attempt to completely replace the global model.
    """
    attacked_dict = {}
    for key, tensor in delta_dict.items():
        if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
            attacked_dict[key] = tensor * target_scale
        else:
            attacked_dict[key] = tensor.clone()
    return attacked_dict
