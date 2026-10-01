import pytest
import torch
from torch_geometric.data import HeteroData
from src.models.hetero_graphsage import create_hetero_graphsage
from src.models.init import initialize_model

def test_initialize_model():
    metadata = (['transaction', 'account'], [('transaction', 'receives', 'account'), ('account', 'sends', 'transaction')])
    model = create_hetero_graphsage(metadata, hidden_channels=16, out_channels=1, num_layers=2)
    
    # Model should have uninitialized parameters natively
    # Let's create dummy data
    data = HeteroData()
    data['transaction'].x = torch.randn(10, 4)
    data['account'].x = torch.randn(5, 4)
    data['transaction', 'receives', 'account'].edge_index = torch.tensor([[0, 1], [1, 2]])
    data['account', 'sends', 'transaction'].edge_index = torch.tensor([[1, 2], [0, 1]])
    
    initialized = initialize_model(model, data)
    
    # We should be able to get sum of parameters without UninitializedParameter error
    total_params = sum(p.numel() for p in initialized.parameters())
    assert total_params > 0
