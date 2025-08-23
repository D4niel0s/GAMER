import torch
from torch_geometric.data import Data



def text_to_graph(embeds: torch.Tensor, **kwargs):
    """
    Convert given text embeddings to a latent graph representation.
    When given a batch of embeddings (3-dim tensor), will return a batch of graphs.
    Pad tokens are ignored in the resulting graph, according to the provided attention mask.

    Args:
        embeds (torch.Tensor): Tensor of shape (seq_len, d_embed) or (batch, seq_len, d_embed).
        attn_mask (torch.Tensor, optional): Attention mask of shape (seq_len,) or (batch, seq_len), where 0 is a pad token and 1 is a valid token.
    """

    # currently the built graph is a line graph, representing a 'natural' sequence.
    return text_to_line_graph(embeds, **kwargs)


def text_to_line_graph(embeds: torch.Tensor, attn_mask: torch.Tensor = None):
    """
    Convert given text embeddings to a line graph.
    When given a batch of embeddings (3-dim tensor), will return a batch of graphs.
    Pad tokens are ignored in the resulting graph, according to the provided attention mask.

    Args:
        embeds (torch.Tensor): Tensor of shape (seq_len, d_embed) or (batch, seq_len, d_embed).
        attn_mask (torch.Tensor, optional): Attention mask of shape (seq_len,) or (batch, seq_len), where 0 is a pad token and 1 is a valid token.
    """

    def seq_to_line_graph(embeds: torch.Tensor, sample_attn_mask: torch.Tensor = None):
        # Helper for converting a single sample (no batch dim) to a line graph.

        # Remove pad tokens from the resulting graph
        embeds = embeds[sample_attn_mask.bool()] if sample_attn_mask is not None else embeds

        L = embeds.size(0) # seq len

        src_nodes = torch.arange(L-1, device=embeds.device)
        target_nodes = torch.arange(1, L, device=embeds.device)

        # forward edges (i → i+1) and backward edges (i+1 → i)
        src = torch.cat([src_nodes, target_nodes], dim=0)
        target = torch.cat([target_nodes, src_nodes], dim=0)

        edge_index = torch.stack([src, target], dim=0)

        return Data(x=embeds, edge_index=edge_index)



    if embeds.dim() == 2:
        # Single sample, no batch dimension
        if attn_mask is not None and (attn_mask.dim() != 1 or attn_mask.size(0) != embeds.size(0)):
            raise ValueError(f"Attention mask must be 1D with length {embeds.size(0)}, got {attn_mask.size()}.")
            
        return [seq_to_line_graph(embeds, attn_mask)]
    
    elif embeds.dim() != 3:
        raise ValueError(f"Expected 2D or 3D tensor, got {embeds.dim()}D tensor.")
    

    if attn_mask is not None and (attn_mask.dim() != 2 or attn_mask.size(0) != embeds.size(0) or attn_mask.size(1) != embeds.size(1)):
        raise ValueError(f"Attention mask must be 2D with shape ({embeds.size(0)}, {embeds.size(1)}), got {attn_mask.size()}.")
        
    return [seq_to_line_graph(embeds[i], attn_mask[i]) for i in range(embeds.size(0))]