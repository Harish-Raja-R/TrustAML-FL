import torch
import torch.nn.functional as F
try:
    from torch_geometric.nn import SAGEConv, to_hetero
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    
class BaseGraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric not found.")
        self.convs = torch.nn.ModuleList()
        # For a hetero model, we use lazy initialization for the first layer (-1)
        self.convs.append(SAGEConv((-1, -1), hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv((-1, -1), hidden_channels))
            
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = self.lin(x)
        return torch.sigmoid(x).squeeze(-1)

def create_hetero_graphsage(metadata, hidden_channels=64, out_channels=1, num_layers=2):
    """
    metadata: data.metadata() from the HeteroData object
    """
    if not PYG_AVAILABLE:
        raise ImportError("PyTorch Geometric not found.")
    model = BaseGraphSAGE(hidden_channels, out_channels, num_layers)
    model = to_hetero(model, metadata, aggr='mean')
    return model
