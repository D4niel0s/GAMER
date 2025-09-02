import warnings, torch, numpy as np
warnings.filterwarnings("ignore")

from torch_geometric.data import Data
from collections import deque
from primefac import primefac



################################
#### Basic Cayley Builders #####
################################

def get_minimal_n_cayley_graph(num_nodes: int):
    '''Returns the edge index of a minimal Cayley graph, given |V| = n.'''
    n = get_cayley_n(num_nodes)
    return get_cayley_graph(n)


def get_cayley_graph(n: int): 
    '''Get the edge index of the Cayley graph (Cay(SL(2, Z_n); S_n)).'''
    generators = np.array([
        [[1, 1], [0, 1]],
        [[1, n-1], [0, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [n-1, 1]]
    ])
    ind = 1

    queue = deque([np.array([[1, 0], [0, 1]])])
    nodes = {(1, 0, 0, 1): 0}

    senders = []
    receivers = []

    while queue:
        x = queue.pop()
        x_flat = (x[0][0], x[0][1], x[1][0], x[1][1])
        assert x_flat in nodes
        ind_x = nodes[x_flat]
        for i in range(4):
            tx = np.matmul(x, generators[i])
            tx = np.mod(tx, n)
            tx_flat = (tx[0][0], tx[0][1], tx[1][0], tx[1][1])
            if tx_flat not in nodes:
                nodes[tx_flat] = ind
                ind += 1
                queue.append(tx)
            ind_tx = nodes[tx_flat]

            senders.append(ind_x)
            receivers.append(ind_tx)
    return torch.tensor([senders, receivers])


def get_cayley_n(num_nodes: int) -> int:
    '''Finds the minimal n-parameter of a Cayley graph'''
    n = 1
    while cayley_graph_size(n) < num_nodes:
        n += 1
    return n


def cayley_graph_size(n: int) -> int:
    '''Returns the number of nodes in a Cayley graph, given its n-parameter'''
    n = int(n)
    return round(n*n*n*np.prod([1 - 1.0/(p * p) for p in list(set(primefac(n)))]))



################################
#### Complex Cayley Builder ####
################################

def build_cayley_graph(text_embeds: torch.Tensor,
                    image_embeds: torch.Tensor,
                    attn_mask: torch.Tensor
                    ) -> list[Data]:
     
	if text_embeds.shape[0] != attn_mask.shape[0]:
		raise ValueError("text_embeds and attn_mask must have same batch size")
    
	res = []
	for i in range(text_embeds.shape[0]): # For each sample in batch, build Cayley Graph with embeddings.
		text_emb = text_embeds[i][attn_mask[i].bool()] if attn_mask is not None else text_embeds[i]
		img_emb = image_embeds[i]

		feats = torch.cat([text_emb, img_emb], dim=0)
		n_nodes = feats.shape[0]

		cayley_graph = get_minimal_n_cayley_graph(n_nodes)
		num_virts = cayley_graph_size(get_cayley_n(n_nodes)) - n_nodes # Number of additional nodes.

		virt_feats = torch.zeros((num_virts, text_emb.shape[1]), device=text_embeds.device)

		res.append(
			Data(x=torch.cat([feats, virt_feats], dim=0), edge_index=cayley_graph)
		)
	
	return res