import torch, torch.nn as nn

from typing import Optional, Literal
from torch_geometric.nn import (
    GPSConv,
    GINEConv,
    GINConv,
    global_mean_pool,
    LayerNorm,
)

from .utils import mlp



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
        self.node_mlp = mlp(ch_in=ch_in, ch_out=hidden_dim, hidden=hidden_dim, num_hidden_layers=1, dropout=dropout)

        # Projection for edge feats to hidden dimension (if edge_dim > 0)
        self.edge_mlp = mlp(edge_dim, hidden_dim, hidden_dim, num_hidden_layers=1, dropout=dropout) if edge_dim > 0 else None
        

        # Stack of GPSConv blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):

            if edge_dim > 0:
                local_gnn = GINEConv(
                    mlp(hidden_dim, hidden_dim, hidden_dim, num_hidden_layers=1, dropout=dropout),
                    train_eps=True,
                    edge_dim=hidden_dim
                )
            else:
                local_gnn = GINConv(
                    mlp(hidden_dim, hidden_dim, hidden_dim, num_hidden_layers=1, dropout=dropout),
                    train_eps=True
                )


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
            hidden = int(post_mlp_width_mult * hidden_dim),
            num_hidden_layers = 0,
            dropout = dropout
        )

        # Post aggregation readout
        self.readout = mlp(
            ch_in = hidden_dim,
            ch_out = num_tasks,
            hidden = hidden_dim,
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
            h = layer(x, edge_index, batch=batch, edge_attr=edge_attr)
            x = x + h
            
        # postnet + residual
        h = self.postnet(x)
        x = x + h

        # Graph-level pooling
        g = global_mean_pool(x, batch)

        # Predict
        out = self.readout(g)
        return out