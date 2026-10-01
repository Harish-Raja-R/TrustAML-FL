import pytest
import torch
from src.robustness.attacks import sign_flip_attack, model_replacement_attack

def test_sign_flip_attack():
    delta = {
        "layer1.weight": torch.tensor([1.0, -2.0, 3.0]),
        "layer1.bias": torch.tensor([0.5])
    }
    attacked = sign_flip_attack(delta)
    
    assert torch.allclose(attacked["layer1.weight"], torch.tensor([-1.0, 2.0, -3.0]))
    assert torch.allclose(attacked["layer1.bias"], torch.tensor([-0.5]))
    
def test_model_replacement_attack():
    delta = {
        "layer1.weight": torch.tensor([1.0, -2.0, 3.0]),
        "layer1.bias": torch.tensor([0.5])
    }
    attacked = model_replacement_attack(delta, target_scale=10.0)
    
    assert torch.allclose(attacked["layer1.weight"], torch.tensor([10.0, -20.0, 30.0]))
    assert torch.allclose(attacked["layer1.bias"], torch.tensor([5.0]))
