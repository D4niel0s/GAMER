import torch
from torch_geometric.data import Data



def image_to_graph(embeds: torch.Tensor, **kwargs):
    """
    Convert given image patch embeddings to a latent graph representation.
    When given a batch of embeddings (3-dim tensor), will return a batch of graphs.
    Assumes number of patches is a perfect square.

    Args:
        embeds (torch.Tensor): Tensor of shape (n_patches, d_embed) or (batch, n_patches, d_embed).
    """

    # currently the built graph is a grid graph, representing a 'natural' image.
    return image_to_grid_graph(embeds, **kwargs)



def image_to_grid_graph(embeds: torch.Tensor, self_loops: bool = False):
    """
    Convert given image patch embeddings to a grid graph.
    When given a batch of embeddings (3-dim tensor), will return a batch of graphs.
    Assumes number of patches is a perfect square.

    Args:
        embeds (torch.Tensor): Tensor of shape (n_patches, d_embed) or (batch, n_patches, d_embed).
        self_loops (bool, optional): Whether to include self-loops in the graph.
    """


    def get_grid_edges(H,W, self_loops):
        # Returns edge list of neighbors. Since all samples are same shape, we can use the same edge list for all.
        
        
        offsets = [
            [-1,-1], [-1,0], [-1,1],
            [ 0,-1],         [0,1],
            [ 1,-1], [ 1,0], [1,1]
        ]
        if self_loops:
             offsets.append([0,0])

        offsets = torch.tensor(offsets)


        coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1)
        coords = coords.view(-1, 2)  # all possible coordinates in grid (H*W, 2)

        coords_exp = coords[:, None, :]         # exapnd coords along 2nd dim
        offsets_exp = offsets[None, :, :]       # expand offsets along 1st dim
        neighbors = coords_exp + offsets_exp    # sum to get neighbors of all nodes [H*W, num_neighbors, 2]

        valid_mask = (neighbors[...,0]>=0) & (neighbors[...,0]<H) & \
                     (neighbors[...,1]>=0) & (neighbors[...,1]<W)       # neighbors within bounds

        src = torch.arange(H*W).view(-1,1).expand(-1,offsets.shape[0])[valid_mask]
        dst = (neighbors[...,0]*W + neighbors[...,1])[valid_mask]

        return torch.stack([src, dst], dim=0).to(embeds.device)

    if embeds.dim() == 2:
            # Single sample, no batch dimension
            embeds = embeds.unsqueeze(0)
    if embeds.dim() != 3:
        raise ValueError(f"Expected 2D or 3D tensor, got {embeds.dim()}D tensor.")

    B, num_patches, _ = embeds.shape
    H = W = int(num_patches ** 0.5)
    assert H * W == num_patches, "n_patches must be a perfect square"


    edge_index = get_grid_edges(H, W, self_loops)
    return [Data(x=embeds[i], edge_index=edge_index.clone()) for i in range(B)]