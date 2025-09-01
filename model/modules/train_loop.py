import warnings, torch, requests
warnings.filterwarnings("ignore")

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


from mmg_builder import build_multimodal_graph 				# embeds -> graph
from utils import preprocess_dataset, batch_to_model_inputs # graph -> batch + PE
from graphGPS import GraphGPSNet 							# model

from datasets import load_from_disk

# Assume dataset has column 'multimodal_graph' which contains dict with x,edge_index,edge_attrs
data_dir = '/home/yandex/MLWG2025/danielvolkov/datasets/VQA_mmg_BERT_BeiT_6fus_2tg/'
data = load_from_disk(data_dir)

sample_graph = {k: torch.tensor(v) for k, v in data['train'][0]['multimodal_graph'].items()}
print({k: v.shape for k, v in sample_graph.items()})

sample = [Data(**sample_graph)]
print(sample)



model = GraphGPSNet(
	node_dim = 768 + 5, # 768 is BERT/BeiT embeds, 5 extra feats are node type one-hots
	edge_dim = 6, 		# 6 feats are edge type one-hots
	num_tasks = 3_000,	# For VQA we need to predict from 3,000 classes (answers)
	hidden_dim = 768,	# idk man just match BERT or smth seems legit
	pe_dim = 16,		# standard (I thinks?)
	num_layers = 5,	    # Graph is connected with max 5 hop distance
	heads = 8,         	# Random ass number that seems cool
	dropout = 0.1,		# idk man wtf is dropout
	post_mlp_width_mult = 2.0,
	readout_method = 'mean'
)

print(model)

num_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {num_params: ,}")

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {num_trainable_params: ,}")



loader = DataLoader(sample) # batch_size=10, shuffle=True
loader_with_pe = preprocess_dataset(loader)

next(iter(loader_with_pe))



input_batch = batch_to_model_inputs(next(iter(loader_with_pe)))
print(input_batch)


model(**input_batch) # Prints actual numbers huzzah!!
