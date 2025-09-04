import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt, networkx as nx, torch.nn.functional as F

from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Linear
from collections import Counter
from typing import Iterable
from tqdm import tqdm


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
    dataset = [transform(data) for data in tqdm(dataset)]
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


def vqa_answers_to_soft_label(answers, ans2idx):
    """
    Convert VQA answers into a soft target vector.

    Args:
        answers (list[dict]): list of dicts with at least an "answer" field
        ans2idx (dict): mapping from answer string -> class index

    Returns:
        torch.FloatTensor: (len(ans2idx),) soft target vector
    """

    target = torch.zeros(len(ans2idx), dtype=torch.float32)

    # Count how many annotators gave each answer
    counts = Counter([a["answer"] for a in answers])

    for ans, cnt in counts.items():
        if ans in ans2idx:  # skip OOV answers
            target[ans2idx[ans]] = min(cnt / 3.0, 1.0) # This is the VQA protocol

    return target

def vqa_score_from_soft_targets(logits, soft_targets):
    """
    Compute VQA accuracy score from logits and soft targets.

    Args:
        logits (torch.Tensor): (B, C) raw model outputs
        soft_targets (torch.Tensor): (B, C) soft targets from VQA protocol

    Returns:
        float: mean VQA accuracy over batch
    """
    
    preds = torch.argmax(logits, dim=-1)

    # Hacky gather() trick. Just take soft target that matches the predicted class.
    scores = soft_targets.gather(1, preds.unsqueeze(1)).squeeze(1)

    return scores.mean().item()




# Cayley Utils
# code based on: https://github.com/eemlcommunity/PracticalSessions2024/tree/main/4_geometric_deep_learning
def draw_one_graph(ax, edges, label=None, node_emb=None, layout=None,
                   special_color=False, pos=None):
    """draw a graph with networkx based on adjacency matrix (edges)
    graph labels could be displayed as a title for each graph
    node_emb could be displayed in colors
    """
    graph = nx.Graph()
    edges = zip(edges[0], edges[1])
    graph.add_edges_from(edges)
    if layout == 'custom':
      node_pos = pos
    elif layout == 'tree':
      node_pos=nx.nx_agraph.graphviz_layout(graph, prog='dot')
    else:
      node_pos = layout(graph)
    #add colors according to node embeding
    if (node_emb is not None) or special_color:
        color_map = []
        node_list = [node[0] for node in graph.nodes(data = True)]
        for i,node in enumerate(node_list):
            #just ignore this branch
            if special_color:
                if len(node_list) == 3:
                    crt_color = (1,0,0)
                elif len(node_list) == 5:
                    crt_color = (0,1,0)
                elif len(node_list) == 4:
                    crt_color = (1,1,0)
                else:
                  special_list = [(1,0,0)] * 3 + [(0,1,0)] * 5 + [(1,1,0)] * 4
                  crt_color = special_list[i]
            else:
                crt_node_emb = node_emb[node]
                #map float number (node embeding) to a color
                crt_color = cm.gist_rainbow(crt_node_emb, bytes=True)
                crt_color = (crt_color[0]/255.0, crt_color[1]/255.0, crt_color[2]/255.0, crt_color[3]/255.0)
            color_map.append(crt_color)

        nx.draw_networkx_nodes(graph,node_pos, node_color=color_map,
                        nodelist = node_list, ax=ax)
        nx.draw_networkx_edges(graph, node_pos, ax=ax)
        nx.draw_networkx_labels(graph,node_pos, ax=ax)
    else:
        nx.draw_networkx(graph, node_pos, ax=ax)

# code based on: https://github.com/eemlcommunity/PracticalSessions2024/tree/main/4_geometric_deep_learning
def gallery(graphs, labels=None, node_emb=None, special_color=False, max_graphs=4, max_fig_size=(40, 10), layout=nx.layout.kamada_kawai_layout):
    ''' Draw multiple graphs as a gallery
    Args:
      graphs: torch_geometrics.dataset object/ List of Graph objects
      labels: num_graphs
      node_emb: num_graphs* [num_nodes x num_ch]
      max_graphs: maximum graphs display
    '''
    num_graphs = min(len(graphs), max_graphs)
    ff, axes = plt.subplots(1, num_graphs,
                            figsize=max_fig_size,
                            subplot_kw={'xticks': [], 'yticks': []})
    if num_graphs == 1:
        axes = [axes]
    if node_emb is None:
        node_emb = num_graphs*[None]
    if labels is None:
        labels = num_graphs * [" "]


    for i in range(num_graphs):
        draw_one_graph(axes[i], graphs[i].edge_index.numpy(), labels[i], node_emb[i], layout, special_color,
                            pos=graphs[i].pos)
        if labels[i] != " ":
            axes[i].set_title(f"Target: {labels[i]}", fontsize=28)
        axes[i].set_axis_off()
    plt.show()