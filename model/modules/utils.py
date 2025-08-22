import torch, torch.nn as nn

from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Linear
from typing import Iterable



# ---- ML utils ---- #


def mlp(
        ch_in: int,
        ch_out: int,
        hidden: int,
        num_hidden_layers: int = 1,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU
):
    """
    Build an MLP with `num_hidden_layers` hidden layers, as-well as input and output layers.
    Each hidden layer has `hidden` width, uses the specified activation function, and uses dropout with the specified rate.
    
    The MLP structure is as such:
        - Linear(`ch_in`, `hidden`), activation, dropout
        - `num_hidden_layers` * (Linear(`hidden`,`hidden`), activation, dropout)
        - Linear(`hidden`, `ch_out`)

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



# ---- Graph utils ---- #


def union_graph(graphs: list[Data]) -> Data:
    """
    Return a union graph of a given list of graphs.
    Each graph must have the same node feature dimension.

    Args:
        graphs (list[Data]): List of graphs to union.
    """

    if not graphs: # Goofy edge case
        return Data()

    # Initialize with the first graph
    union_graph = graphs[0].clone()
    total_nodes = union_graph.num_nodes

    for graph in graphs[1:]:
        offset = total_nodes
        # Add graph's nodes
        union_graph.x = torch.cat([union_graph.x, graph.x], dim=0)

        # Offset new edge indices by the number of nodes in the union graph so far (match the indices)
        new_edges = graph.edge_index + offset

        union_graph.edge_index = torch.cat([union_graph.edge_index, new_edges], dim=1)
        total_nodes += graph.num_nodes

    return union_graph

def add_nodes(data: Data, num_new: int) -> torch.Tensor:
    """
    Add `num_new` nodes to the given graph and return their indices in the resulting graph.
    New nodes' features are initialized to zero vectors.
    Edges of the graph remain unchanged.

    Args:
        data (Data): The graph to which nodes will be added.
        num_new (int): The number of new nodes to add.
    """

    old = data.num_nodes
    new_idx = torch.arange(old, old + num_new, device=data.x.device)
    data.x = torch.cat([data.x, torch.zeros(num_new, data.x.size(-1), device=data.x.device)], dim=0)
    return new_idx




# ---- Data / Batch utils ---- #


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