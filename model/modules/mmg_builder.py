import torch
from torch_geometric.data import Data
from txt2graph import text_to_graph
from img2graph import image_to_graph





def build_multimodal_graph(text_embeds: torch.Tensor,
                           image_embeds: torch.Tensor,
                           attn_mask: torch.Tensor,
                           self_loops: bool = False,
                           grid_size: int = 14,
                           ):
    
    """
    Build a graph representation of multimodal data (text & image).
    
    Currently:
        The text is converted to a line-graph, and the image is converted to a grid graph.
        The resulting graphs are merged into a single graph, adding virtual nodes in a hierarchical manner (4 virts for quadrants of image, 1 text-global, and 1 fusion virt).

        A square grid of size N x N is assumed for the image graph, where N is the grid_size parameter.

    Args:
        text_embeds (torch.Tensor): Text embeddings of shape (batch_size, seq_len, d_embed).
        image_embeds (torch.Tensor): Image embeddings of shape (batch_size, n_patches, d_embed).
        attn_mask (torch.Tensor): Attention mask for text embeddings of shape (batch_size, seq_len), where 1 indicates valid tokens and 0 indicates padding.
        self_loops (bool): Whether to add self-loops to the image graph. Default: False.
        grid_size (int): Size of the grid for the image graph. Default: 14.
    """

    if text_embeds.shape[0] != attn_mask.shape[0]:
        raise ValueError("text_embeds and attn_mask must have same batch size")

    text_graphs = text_to_graph(text_embeds, attn_mask=attn_mask)
    image_graphs = image_to_graph(image_embeds, self_loops=self_loops)

    return [build_hierarchical_mmg(i_graph, t_graph, N=grid_size) for i_graph, t_graph in zip(image_graphs, text_graphs)]





def build_hierarchical_mmg(image_graph: Data, text_graph: Data, N: int=14) -> Data:
    """
    Fuse a text (line) graph and an image (grid) graph into a multimodal graph.
    The resulting graph will have:
      - 4 quadrant virtual nodes (K4 + connected to their pixels)
      - 1 text-global virtual node (connected to all text nodes)
      - 1 fusion virtual node (connected to quadrants + text-global)

    
    A square grid of size N x N is assumed for the image graph.

    Args:
        image_graph (Data): Image graph, assumed to be a grid graph.
        text_graph (Data): Text graph, assumed to be a line graph.
        N (int): Size of the grid for the image graph. Default: 14.
    """

    data = union_graph([image_graph, text_graph])


    # Calculate quadrant masks
    coords = torch.stack(torch.meshgrid(
        torch.arange(N, device=data.x.device), 
        torch.arange(N, device=data.x.device), 
        indexing='ij'
    ), dim=-1).reshape(-1, 2)  # (N*N, 2)
    
    mid = N // 2
    masks = [
        (coords[:,0] < mid) & (coords[:,1] < mid),   # top-left
        (coords[:,0] < mid) & (coords[:,1] >= mid),  # top-right
        (coords[:,0] >= mid) & (coords[:,1] < mid),  # bottom-left
        (coords[:,0] >= mid) & (coords[:,1] >= mid), # bottom-right
    ]

    # Add quadrant nodes and edges between them and their patches
    q_nodes = add_nodes(data, 4)

    for i, mask in enumerate(masks):
        node_idxs = torch.nonzero(mask, as_tuple=False).view(-1)
        q = q_nodes[i].repeat(node_idxs.size(0))
        e1 = torch.stack([q, node_idxs])
        e2 = torch.stack([node_idxs, q])
        data.edge_index = torch.cat([data.edge_index, e1, e2], dim=1)

    # K4 between quadrants
    q1, q2 = torch.meshgrid(q_nodes, q_nodes, indexing="ij")
    mask = q1 != q2
    e = torch.stack([q1[mask], q2[mask]])
    data.edge_index = torch.cat([data.edge_index, e], dim=1)


    # Add text-global node and edges between it and all text nodes ( K_{1,|text|} )
    text_global = add_nodes(data, 1)[0]
    text_start = image_graph.num_nodes

    text_idxs = torch.arange(text_start, text_start + text_graph.num_nodes, device=data.x.device)
    tg = text_global.repeat(text_idxs.size(0))
    e1 = torch.stack([tg, text_idxs])
    e2 = torch.stack([text_idxs, tg])
    data.edge_index = torch.cat([data.edge_index, e1, e2], dim=1)

    # Add fusion node and edges between it and all special nodes (quadrants + text-global, resulting in K_{1,5} )
    fusion = add_nodes(data, 1)[0]
    specials = torch.cat([q_nodes, text_global.view(-1)])
    f = fusion.repeat(specials.size(0))
    e1 = torch.stack([f, specials])
    e2 = torch.stack([specials, f])
    data.edge_index = torch.cat([data.edge_index, e1, e2], dim=1)

    return data



def union_graph(graphs: list[Data]) -> Data:
    """
    Return a union graph of a given list of graphs.
    Each graph must have the same edge feature dimension.

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
    Add num_new nodes to the given graph and return their indices in the resulting graph.
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