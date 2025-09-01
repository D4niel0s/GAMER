import warnings, torch, requests
warnings.filterwarnings("ignore")

from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data

from utils import preprocess_dataset, batch_to_model_inputs # graph -> batch + PE
from graphGPS import GraphGPSNet 							# model

from datasets import load_from_disk



def row_to_data(row):
    x = torch.tensor(row["x"], dtype=torch.float)
    edge_index = torch.tensor(row["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(row["edge_attr"], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class GraphDataset(Dataset):
    def __init__(self, hf_dataset, column="multimodal_graph", transform=None):
        super().__init__(None, transform)
        self.hf_dataset = hf_dataset
        self.column = column

    def len(self):
        return len(self.hf_dataset)

    def get(self, idx):
        return row_to_data(self.hf_dataset[idx][self.column])


def load_data_add_pe(data_dir):
    dataset = load_from_disk(data_dir)

    train_dl = DataLoader(GraphDataset(dataset["train"]), batch_size=32, shuffle=True)
    val_dl = DataLoader(GraphDataset(dataset["validation"]), batch_size=32, shuffle=False)

    return  preprocess_dataset(train_dl), preprocess_dataset(val_dl)

# Assume dataset has column 'multimodal_graph' which contains dict with x,edge_index,edge_attrs
data_dir = '/home/yandex/MLWG2025/danielvolkov/datasets/VQA_mmg_BERT_BeiT_6fus_2tg/'


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


train_dl, val_dl = load_data_add_pe(data_dir)


input_batch = batch_to_model_inputs(next(iter(train_dl)))
print(input_batch)

res = model(**input_batch) # Prints actual numbers huzzah!!
print(f'{res=}, {res.shape=}')
