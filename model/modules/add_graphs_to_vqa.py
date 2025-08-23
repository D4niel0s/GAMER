from transformers import BeitImageProcessor, BeitModel, BertTokenizer, BertModel
from PIL import Image

from mmg_builder import build_multimodal_graph 				# embeds -> graph
from datasets import load_dataset
import torch



bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").eval().cuda()

beit_processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
beit = BeitModel.from_pretrained('microsoft/beit-base-patch16-224', use_safetensors=True).eval().cuda()



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
	graphs = build_multimodal_graph(text_embeds=q_embs,
									image_embeds=i_embs,
									attn_mask=attn_mask,
									self_loops=False,
									fusion_num=6,
									text_global_num=2
	)

	batch['multimodal_graph'] = [data_to_dict(graph) for graph in graphs]

	return batch



def data_to_dict(data):
    """Convert PyTorch Geometric Data to serializable dict"""
    return {
        'x': data.x if data.x is not None else None,
        'edge_index': data.edge_index if data.edge_index is not None else None,
        'edge_attr': data.edge_attr if data.edge_attr is not None else None,
        # Add other attributes as needed
    }

def dict_to_data(data_dict):
    """Convert dict back to PyTorch Geometric Data"""
    from torch_geometric.data import Data
    import torch
    
    return Data(
        x=torch.tensor(data_dict['x']) if data_dict['x'] is not None else None,
        edge_index=torch.tensor(data_dict['edge_index']) if data_dict['edge_index'] is not None else None,
        edge_attr=torch.tensor(data_dict['edge_attr']) if data_dict['edge_attr'] is not None else None,
    )




vqa_data = load_dataset("pingzhili/vqa_v2", split="train")
print("Hooray!! VQA downloaded!! Starting processing...")


encoded = vqa_data.map(
    process_batch,
    batched=True,
    batch_size=64,
)
print("Huzzah!! Processing complete! Saving to disk...")


encoded.save_to_disk('../../data/VQA/BERT_BeiT_Hierarchical_6F2TG/train')
print("Done!")