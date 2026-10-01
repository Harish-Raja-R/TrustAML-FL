import torch

def sign_flip_attack(state_dict, scale=-1.0):
    """
    Inverts the gradients (pseudo-gradients) of the client.
    Requires passing the pseudo-gradient rather than raw weights,
    or we just invert the weights (which destroys the model).
    Let's invert the difference from global model.
    """
    attacked_dict = {}
    for key, tensor in state_dict.items():
        if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
            attacked_dict[key] = tensor * scale
        else:
            attacked_dict[key] = tensor.clone()
    return attacked_dict

def gaussian_attack(state_dict, std=1.0):
    """
    Adds massive Gaussian noise to the weights.
    """
    attacked_dict = {}
    for key, tensor in state_dict.items():
        if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
            attacked_dict[key] = tensor + torch.randn_like(tensor) * std
        else:
            attacked_dict[key] = tensor.clone()
    return attacked_dict
