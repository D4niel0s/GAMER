import os
import torch
os.environ['TORCH'] = torch.__version__

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import networkx as nx
from functools import reduce
from tqdm import tqdm
import primefac
from utils import draw_one_graph, gallery
from collections import deque
from primefac import primefac
import warnings, torch, requests, matplotlib.pyplot as plt, networkx as nx
from transformers import BeitImageProcessor, BeitModel, BertTokenizer, BertModel
from torch_geometric.utils import to_networkx
from PIL import Image
warnings.filterwarnings("ignore")

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
		text_emb = text_embeds[i][attn_mask[i].bool()]
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


def process_batch(batch):
	questions = batch['question']
	image_ids = batch['image_id']

	# Embed question
	q_embs, attn_mask = embed_text(questions)

	# dedup while keeping map
	id2img = {}
	for idx, img_id in enumerate(image_ids):
		if img_id not in id2img:
			id2img[img_id] = batch['image'][idx].convert("RGB")

	unique_embeds = embed_image(list(id2img.values()))

	id_to_emb = {k: v for k, v in zip(id2img.keys(), unique_embeds)}
	i_embs = torch.stack([id_to_emb[img_id] for img_id in image_ids])

	# Build multimodal graphs
	graphs = build_cayley_graph(text_embeds=q_embs,
								image_embeds=i_embs,
								attn_mask=attn_mask
	)

	batch['multimodal_graph'] = [data_to_dict(graph) for graph in graphs]

	return batch


def embed_text(texts: list[str]):
    inputs = bert_tokenizer(texts, return_tensors="pt", padding=True).to('cuda')
    with torch.no_grad():
        outputs = bert(**inputs)
    return outputs.last_hidden_state, inputs['attention_mask'].to('cuda')


def embed_image(images: list[Image.Image]):
    inputs = beit_processor(images, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = beit(**inputs)
    return outputs.last_hidden_state[:, 1:, :]


def data_to_dict(data):
    """Convert PyTorch Geometric Data to Arrow/NumPy-friendly dict"""
    return {
        "x": data.x.cpu().numpy().astype("float32") if data.x is not None else None,
        "edge_index": data.edge_index.cpu().numpy().astype("int64") if data.edge_index is not None else None,
    }


################################
######## Demonstration #########
################################

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")


model_name = "microsoft/beit-base-patch16-224"
beit_processor = BeitImageProcessor.from_pretrained(model_name)
beit = BeitModel.from_pretrained(model_name, use_safetensors=True)


# text = ["BERT is great for natural language processing!"]
text = ["BERT is great for natural language processing!", "Hi my name is Danik.", 
        "Lorem ipsum dolor sit amet consectetur adipiscing elit quisque faucibus ex \
        sapien vitae pellentesque sem placerat in id cursus mi pretium tellus \
        duis convallis tempus leo eu aenean sed diam urna tempor pulvinar vivamus \
        fringilla lacus nec metus bibendum egestas iaculis massa nisl malesuada lacinia \
        integer nunc posuere ut hendrerit semper vel class aptent taciti sociosqu ad \
        litora torquent per conubia nostra inceptos himenaeos orci varius natoque penatibus \
        et magnis dis parturient montes nascetur ridiculus mus donec rhoncus eros lobortis \
        nulla molestie mattis scelerisque maximus eget fermentum odio phasellus non purus est \
        efficitur laoreet mauris pharetra vestibulum fusce dictum risus."]
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
text_inputs = bert_tokenizer(text, return_tensors="pt", padding=True)
# Forward pass (no gradient calculation needed for inference)
with torch.no_grad():
    outputs = bert(**text_inputs)
# Extract embeddings
text_embeds = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]
print(f'{text_embeds.shape=}')

img_inputs = beit_processor(images=[image]*3, return_tensors="pt")
print(f'{img_inputs["pixel_values"].shape=}')

with torch.no_grad():
    outputs = beit(**img_inputs, output_hidden_states=True)

# outputs.last_hidden_state → [batch_size, num_patches+1, hidden_dim]
# The first token is [CLS], the rest are patch embeddings
last_hidden = outputs.last_hidden_state

img_embeds = last_hidden[:, 1:, :]  # contextualized patch embeddings
print(f"Patches shape: {img_embeds.shape}")  # (1, num_patches, hidden_dim)
res = build_cayley_graph(
    text_embeds=text_embeds,
    image_embeds=img_embeds,
    attn_mask=text_inputs['attention_mask'],
)
gallery(res)