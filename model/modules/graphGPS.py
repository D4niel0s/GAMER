import torch, torch.nn as nn

from torch_geometric.transforms import AddLaplacianEigenvectorPE
from typing import Optional, Iterable, Literal
from torch_geometric.data import Batch
from torch_geometric.nn import (
    GPSConv,
    GINEConv,
    global_mean_pool,
    Linear,
    LayerNorm,
)


def mlp(
        ch_in: int,
        ch_out: int,
        hidden: int,
        num_hidden_layers: int = 1,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU
):
    """
    Build an MLP with 'num_hidden_layers' hidden layers, as-well as input and output layers.
    Each hidden layer has 'hidden' width, uses the specified activation function, and uses dropout with the specified rate.
    The MLP structure is as such:
        - Linear(ch_in, hidden), activation, dropout
        - num_hidden_layers * (Linear(hidden,hidden), activation, dropout)
        - Linear(hidden, ch_out)

    Args:
        ch_in (int): Input feature dimension.
        ch_out (int): Output feature dimension.
        hidden (int): Width of hidden layers.
        num_hidden_layers (int): Number of hidden layers (default: 2).
        dropout (float): Dropout rate (default: 0.0).
        act (nn.Module): Activation function to use (default: nn.GELU).
    """

    layers = [Linear(ch_in, hidden), act(), nn.Dropout(dropout)]

    for _ in range(num_hidden_layers):
        layers += [Linear(hidden, hidden), act(), nn.Dropout(dropout)]

    layers += [Linear(hidden, ch_out)]

    return nn.Sequential(*layers)




class GraphGPSNet(nn.Module):
    def __init__(
        self, *,                          # makes everything after pass as a kwarg
        node_dim: int,                    # node feature dim (0 if none)
        edge_dim: int = 0,                # edge feature dim (0 if none)
        num_tasks: int = 1,               # graph-level outputs
        hidden_dim: int = 128,
        pe_dim: int = 16,                 # LapPE dimensions (0 to disable)
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        post_mlp_width_mult: int = 2.0,   # hidden width multiplier
        readout_method: Literal['mean'] = 'mean' # TODO: think about readout methods
    ):
        super().__init__()

        self.use_pe = pe_dim > 0
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        self.readout_method = readout_method if readout_method in ['mean'] else None
        assert self.readout_method is not None, "Invalid readout method specified. Use 'mean'."

        # Input projection for node features, with optional PE
        ch_in = node_dim + pe_dim if self.use_pe else node_dim
        self.node_mlp = mlp(ch_in=ch_in, ch_out=hidden_dim, hidden_dim=hidden_dim, num_hidden_layers=1, dropout=dropout)

        # Projection for edge feats to hidden dimension (if edge_dim > 0)
        self.edge_mlp = mlp(edge_dim, hidden_dim, hidden_dim, num_hidden_layers=1, dropout=dropout) if edge_dim > 0 else None
        

        # Stack of GPSConv blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):

            local_gnn = GINEConv(mlp(hidden_dim, hidden_dim, hidden_dim, num_hidden_layers=1, dropout=dropout),
                                    train_eps=True, edge_dim=hidden_dim if edge_dim > 0 else None)
            block = GPSConv(
                channels=hidden_dim,
                conv=local_gnn,     # local message passing
                heads=heads,        # global attention heads
                dropout=dropout,
                act='gelu',
                norm=LayerNorm(hidden_dim)
            )

            self.layers.append(block)

        # Post-encoder feed-forward (like Transformer FFN)
        self.postnet = mlp(
            ch_in = hidden_dim,
            ch_out = hidden_dim,
            hidden_dim = int(post_mlp_width_mult * hidden_dim),
            num_hidden_layers = 0,
            dropout = dropout
        )

        # Post aggregation readout
        self.readout = mlp(
            ch_in = hidden_dim,
            ch_out = num_tasks,
            hidden_dim = hidden_dim,
            num_hidden_layers = 0,
            dropout = dropout
        )


    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        lap_pe: Optional[torch.Tensor] = None,   # provided by AddLaplacianEigenvectorPE
    ):

        # If PE is used, concat PE to node feats
        if self.use_pe:
            x = torch.cat([x, lap_pe], dim=-1)

        # Project node embeds to hidden_dim
        x = self.node_mlp(x)

        # Project edge features to hidden dimension (if they exist)
        edge_attr = self.edge_mlp(edge_attr) if self.edge_dim > 0 else None
        
        # GPS layers (+ residual after each)
        for layer in self.layers:
            x += layer(x, edge_index, batch=batch, edge_attr=edge_attr)

        x += self.postnet(x) # postnet + residual

        # Graph-level pooling
        g = global_mean_pool(x, batch)

        # Predict
        out = self.readout(g)
        return out
    






# TODO: Below is example code from GPT. Still need to work on taking data and running through the model. Pushing this version since it's 0400 AM
def preprocess_dataset(dataset: Iterable, k: int = 16, is_undirected: bool = True):
    """
    Apply Laplacian eigenvector positional encoding (LapPE) to each graph
    in a dataset. Attaches node-level 'lap_pe' to every Data object.
    """
    transform = AddLaplacianEigenvectorPE(k=k, attr_name="lap_pe", is_undirected=is_undirected)
    dataset = [transform(data) for data in dataset]
    return dataset


def batch_to_model_inputs(batch: Batch):
    """
    Convert a PyG Batch into arguments for GraphGPSNet.forward(...).
    """
    return dict(
        x = getattr(batch, "x", None),             # node features or None
        edge_index = batch.edge_index,             # COO edge index
        batch = batch.batch,                       # graph id per node
        edge_attr = getattr(batch, "edge_attr", None),  # optional edge features
        lap_pe = getattr(batch, "lap_pe", None),   # optional LapPE
    )
