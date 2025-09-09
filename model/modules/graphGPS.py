import torch, torch.nn as nn

from typing import Optional, Literal
from torch_geometric.nn import (
    GPSConv,
    GINEConv,
    GINConv,
    global_mean_pool,
    LayerNorm,
    SAGPooling,
    GATConv
)

from .utils import mlp



class GraphGPSNet(nn.Module):
    def __init__(
        self, *,                          # makes everything after pass as a kwarg
        node_dim: int,                    # node feature dim (0 if none)
        edge_dim: int = 0,                # edge feature dim (0 if none)
        output_dim: int = 1,              # Dimension of output vector from model
        hidden_dim: int = 128,
        pe_dim: int = 16,                 # LapPE dimensions (0 to disable)
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        mlps_hidden_layers = 1,

        
        # SAGPool Configuration
        sagpool_mode: Literal['none', 'global', 'hierarchical'] = 'none',
        global_sagpool_ratio: float = 0.6,
        sagpool_layer2ratio: dict[int:float] = {},
        global_pool_method: Literal['mean'] = 'mean' # TODO: think about readout methods
    ):
        super().__init__()

        self.use_pe = pe_dim > 0
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        self.sagpool_mode = sagpool_mode
        self.sagpool_ratios = [sagpool_layer2ratio[i] if i in sagpool_layer2ratio else None for i in range(num_layers)]


        # Input projection for node features, with optional PE
        ch_in = node_dim + pe_dim if self.use_pe else node_dim
        self.node_mlp = mlp(ch_in=ch_in, ch_out=hidden_dim, hidden=hidden_dim, num_hidden_layers=mlps_hidden_layers, dropout=dropout)

        # Projection for edge feats to hidden dimension (if edge_dim > 0)
        self.edge_mlp = mlp(edge_dim, hidden_dim, hidden_dim, num_hidden_layers=mlps_hidden_layers, dropout=dropout) if edge_dim > 0 else None
        

        # Stack of GPSConv blocks
        self.layers = nn.ModuleList()
        self.pools = nn.ModuleList() if self.sagpool_mode == 'hierarchical' else nn.ModuleList([None]*num_layers)

        for i in range(num_layers):

            if edge_dim > 0:
                local_gnn = GINEConv(
                    mlp(hidden_dim, hidden_dim, hidden_dim, num_hidden_layers=mlps_hidden_layers, dropout=dropout),
                    train_eps=True,
                    edge_dim=hidden_dim
                )
            else:
                local_gnn = GINConv(
                    mlp(hidden_dim, hidden_dim, hidden_dim, num_hidden_layers=mlps_hidden_layers, dropout=dropout),
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

            if self.sagpool_mode == 'hierarchical':
                if (ratio := self.sagpool_ratios[i]) is not None:
                    sag_pool = SAGPooling(
                        in_channels=hidden_dim,
                        ratio=ratio,
                        GNN=GATConv,
                        min_score=None,
                        multiplier=1.0
                    )

                else:
                    sag_pool = None

                self.pools.append(sag_pool)

        # Post-encoder feed-forward (like Transformer FFN)
        self.postnet = mlp(
            ch_in = hidden_dim,
            ch_out = hidden_dim,
            hidden =  hidden_dim,
            num_hidden_layers = mlps_hidden_layers,
            dropout = dropout
        )


        if self.sagpool_mode == 'global':
            self.global_sagpool = SAGPooling(
                in_channels=hidden_dim,
                ratio=global_sagpool_ratio,
                GNN=GATConv,
                min_score=None,
                multiplier=1.0
            )
            

        # Post aggregation readout
        self.readout = mlp(
            ch_in = hidden_dim,
            ch_out = output_dim,
            hidden = hidden_dim,
            num_hidden_layers = mlps_hidden_layers,
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
        
        # GPS layers and pools as specified
        for layer, pool in zip(self.layers, self.pools):
            # Apply GPS layer + resid
            if self.edge_dim > 0:
                h = layer(x, edge_index, batch=batch, edge_attr=edge_attr)  # GINE
            else:
                h = layer(x, edge_index, batch=batch)                       # GIN

            x = x + h

            # Apply pool (if exists)
            if pool is not None:
                if self.edge_dim > 0:
                    x, edge_index, edge_attr, batch, _,_ = pool(x, edge_index, edge_attr=edge_attr, batch=batch)
                else:
                    x, edge_index, _, batch, _,_ = pool(x, edge_index, batch=batch)
            
        # postnet + residual
        h = self.postnet(x)
        x = x + h

        if self.sagpool_mode == 'global':
            if self.edge_dim > 0:
                x, edge_index, edge_attr, batch, _,_ = self.global_sagpool(x, edge_index, edge_attr=edge_attr, batch=batch)
            else:
                x, edge_index, _, batch, _,_ = self.global_sagpool(x, edge_index, batch=batch)

        # Graph-level pooling
        g = global_mean_pool(x, batch)

        # Predict
        out = self.readout(g)
        return out