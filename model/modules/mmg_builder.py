import torch
from torch_geometric.data import Data
from txt2graph import text_to_graph
from img2graph import image_to_graph


from utils import union_graph, add_nodes


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

    res = []
    for i_graph, t_graph in zip(image_graphs, text_graphs):
        mmg, info = build_hierarchical_mmg(i_graph, t_graph, N=grid_size)

        res.append(add_type_features(mmg,
                                     num_text = t_graph.num_nodes,
                                     num_image = i_graph.num_nodes,
                                     info = info,
        ))

    return res






EDGE_TYPES = {
        "intra-text": 0,
        "intra-image": 1,
        "img-to-quadrant": 2,
        "quadrant-to-quadrant": 3,
        "text-to-global": 4,
        "fusion-connection": 5,
}

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
    edge_types = []

    # Assign inte text and intra image edge types
    edge_types.extend([EDGE_TYPES["intra-image"]] * image_graph.edge_index.size(1))
    edge_types.extend([EDGE_TYPES["intra-text"]] * text_graph.edge_index.size(1))

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

        edge_types.extend([EDGE_TYPES["img-to-quadrant"]] * (e1.size(1) + e2.size(1)))

    # K4 between quadrants
    q1, q2 = torch.meshgrid(q_nodes, q_nodes, indexing="ij")
    mask = q1 != q2
    e = torch.stack([q1[mask], q2[mask]])
    data.edge_index = torch.cat([data.edge_index, e], dim=1)

    edge_types.extend([EDGE_TYPES["quadrant-to-quadrant"]] * e.size(1))


    # Add text-global node and edges between it and all text nodes ( K_{1,|text|} )
    text_global = add_nodes(data, 1)[0]
    text_start = image_graph.num_nodes

    text_idxs = torch.arange(text_start, text_start + text_graph.num_nodes, device=data.x.device)
    tg = text_global.repeat(text_idxs.size(0))
    e1 = torch.stack([tg, text_idxs])
    e2 = torch.stack([text_idxs, tg])
    data.edge_index = torch.cat([data.edge_index, e1, e2], dim=1)

    edge_types.extend([EDGE_TYPES["text-to-global"]] * (e1.size(1) + e2.size(1)))


    # Add fusion node and edges between it and all special nodes (quadrants + text-global, resulting in K_{1,5} )
    fusion = add_nodes(data, 1)[0]
    specials = torch.cat([q_nodes, text_global.view(-1)])
    f = fusion.repeat(specials.size(0))
    e1 = torch.stack([f, specials])
    e2 = torch.stack([specials, f])
    data.edge_index = torch.cat([data.edge_index, e1, e2], dim=1)

    edge_types.extend([EDGE_TYPES["fusion-connection"]] * (e1.size(1) + e2.size(1)))

    return data, {
        "q_nodes": q_nodes,
        "text_global": text_global,
        "fusion": fusion,
        "edge_types": torch.tensor(edge_types)
    }


def add_type_features(data: Data,
                      num_text: int,
                      num_image: int,
                      info: dict
                      ) -> Data:
    """
    Augments a hierarchical multimodal graph with node-type and edge-type features.
    Adds one-hot encodings of node types to node features (as concatenation to feature vectors), and one-hot encodings of edge types to edge attributes (overriding previous edge attr).

    Assumes that the graph was built by `build_hierarchical_mmg`.

    Args:
        data (Data): The multimodal graph from `build_hierarchical_mmg`.
        num_text (int): Number of text nodes (original, not virtual).
        num_image (int): Number of image nodes (original, not virtual).
        info (dict): The info dictionary returned by `build_hierarchical_mmg`.
    """


    device = data.x.device
    num_nodes = data.num_nodes

    # ---- Node type one-hots ----
    num_node_types = 5
    node_type_feats = torch.zeros((num_nodes, num_node_types), device=device)

    # Assign
    node_type_feats[:num_image, 0] = 1.0                    # image patch
    node_type_feats[num_image:num_image+num_text, 1] = 1.0  # text tokens
    node_type_feats[info['q_nodes'], 2] = 1.0                       # quadrant
    node_type_feats[info['text_global'], 3] = 1.0                   # text-global
    node_type_feats[info['fusion'], 4] = 1.0                        # fusion

    # Concatenate to existing node features
    data.x = torch.cat([data.x, node_type_feats], dim=-1)

    # ---- Edge type one-hots ----
    et = info["edge_types"]  # [E]
    data.edge_attr = torch.nn.functional.one_hot(et, num_classes=len(EDGE_TYPES)).float()


    return data