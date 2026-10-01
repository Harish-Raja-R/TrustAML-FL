import torch

def initialize_model(model, dummy_hetero_data):
    """
    Performs a dummy forward pass to initialize PyTorch Geometric Lazy modules.
    Safe to use before passing the model to federated clients.
    """
    with torch.no_grad():
        if hasattr(dummy_hetero_data, 'x_dict') and hasattr(dummy_hetero_data, 'edge_index_dict'):
            model(dummy_hetero_data.x_dict, dummy_hetero_data.edge_index_dict)
        else:
            # Fallback for normal PyTorch models
            try:
                model(dummy_hetero_data)
            except Exception:
                pass
    return model
