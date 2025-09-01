# import warnings, torch, requests
# warnings.filterwarnings("ignore")

# from torch_geometric.transforms import AddLaplacianEigenvectorPE
# from torch_geometric.loader.dataloader import Collater
# from torch_geometric.data import Dataset, Data
# from torch_geometric.loader import DataLoader

# from utils import preprocess_dataset, batch_to_model_inputs # graph -> batch + PE
# from graphGPS import GraphGPSNet 							# model

# from datasets import load_from_disk



# def row_to_data(row):
#     x = torch.tensor(row["x"], dtype=torch.float)
#     edge_index = torch.tensor(row["edge_index"], dtype=torch.long)
#     edge_attr = torch.tensor(row["edge_attr"], dtype=torch.float)
#     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# def make_transform(k=16, is_undirected=True):
#     return AddLaplacianEigenvectorPE(k=k, attr_name="lap_pe", is_undirected=is_undirected)



# class HFDataset(Dataset):
#     def __init__(self, hf_dataset, column="multimodal_graph", transform=None):
#         super().__init__(None, transform)
#         self.hf_dataset = hf_dataset
#         self.column = column
#         self.transform = transform

#     def len(self):
#         return len(self.hf_dataset)

#     def get(self, idx):
#         row = self.hf_dataset[idx][self.column]
#         data = row_to_data(row)
#         if self.transform is not None:
#             data = self.transform(data)

#         return data

# def load_data_w_pe(data_dir, splits=['train', 'validation'], batch_size=32, shuffle=True):
#     transform = make_transform(k=16)
#     hf_dataset = load_from_disk(data_dir)

#     loaders = []
#     for split in splits:
#         dataset = HFDataset(hf_dataset[split], column="multimodal_graph", transform=transform)
        
#         loader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=shuffle
#         )

#         loaders.append(loader)

#     return tuple(loaders)






# # Assume dataset has column 'multimodal_graph' which contains dict with x,edge_index,edge_attrs
# data_dir = '/home/yandex/MLWG2025/danielvolkov/datasets/VQA_mmg_BERT_BeiT_6fus_2tg/'


# model = GraphGPSNet(
# 	node_dim = 768 + 5, # 768 is BERT/BeiT embeds, 5 extra feats are node type one-hots
# 	edge_dim = 6, 		# 6 feats are edge type one-hots
# 	num_tasks = 3_000,	# For VQA we need to predict from 3,000 classes (answers)
# 	hidden_dim = 768,	# idk man just match BERT or smth seems legit
# 	pe_dim = 16,		# standard (I thinks?)
# 	num_layers = 5,	    # Graph is connected with max 5 hop distance
# 	heads = 8,         	# Random ass number that seems cool
# 	dropout = 0.1,		# idk man wtf is dropout
# 	post_mlp_width_mult = 2.0,
# 	readout_method = 'mean'
# )

# print(model)

# num_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {num_params: ,}")

# num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total number of trainable parameters: {num_trainable_params: ,}")


# train_dl, val_dl = load_data_w_pe(data_dir)

# i=0
# for batch in train_dl:
#     input_batch = batch_to_model_inputs(batch)
#     input_batch = {k: v.to('cuda') for k,v in input_batch.items() if torch.is_tensor(v)}
    
#     res = model(**input_batch) # Prints actual numbers huzzah!!
    
#     if i==100: break
#     i+=1







import warnings, torch
warnings.filterwarnings("ignore")

from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataLoader

from utils import batch_to_model_inputs  # optional, we will implement safe version
from graphGPS import GraphGPSNet
from datasets import load_from_disk
from tqdm import tqdm

# ------------------------
# Helpers
# ------------------------

def row_to_data(row):
    """Convert HF dict to PyG Data."""
    x = torch.tensor(row["x"], dtype=torch.float)
    edge_index = torch.tensor(row["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(row["edge_attr"], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def make_transform(k=16, is_undirected=True):
    return AddLaplacianEigenvectorPE(k=k, attr_name="lap_pe", is_undirected=is_undirected)

# ------------------------
# HF Dataset wrapper
# ------------------------

class HFDataset(Dataset):
    def __init__(self, hf_dataset, column="multimodal_graph", transform=None):
        super().__init__(None, transform)
        self.hf_dataset = hf_dataset
        self.column = column
        self.transform = transform

    def len(self):
        return len(self.hf_dataset)

    def get(self, idx):
        row = self.hf_dataset[idx][self.column]
        data = row_to_data(row)
        if self.transform is not None:
            data = self.transform(data)

        # Detach LapPE and keep on CPU
        if hasattr(data, 'lap_pe'):
            data.lap_pe = data.lap_pe.detach().cpu()
        return data

# ------------------------
# Safe collate_fn
# ------------------------

def collate_fn_no_ptr(batch):
    """
    Collate a list of Data objects into a dict for GraphGPSNet,
    explicitly avoiding ptr and other PyG bookkeeping.
    """
    from torch_geometric.data import Batch

    # Only use fields your model actually expects
    fields = ['x', 'edge_index', 'edge_attr', 'batch']

    # Create PyG Batch to correctly stack edge_index, etc.
    batch_obj = Batch.from_data_list(batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_dict = {}

    for f in fields:
        if hasattr(batch_obj, f):
            batch_dict[f] = getattr(batch_obj, f).to(device)

    return batch_dict


# ------------------------
# Data loaders
# ------------------------

def load_data_w_pe(data_dir, splits=['train', 'validation'], batch_size=32, shuffle=True, k=16):
    hf_dataset = load_from_disk(data_dir)
    transform = make_transform(k=k)

    loaders = []
    for split in splits:
        dataset = HFDataset(hf_dataset[split], column="multimodal_graph", transform=transform)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_fn_no_ptr(batch)
        )
        loaders.append(loader)
    return tuple(loaders)

# ------------------------
# Model
# ------------------------

model = GraphGPSNet(
    node_dim = 768 + 5,
    edge_dim = 6,
    num_tasks = 3000,
    hidden_dim = 768,
    pe_dim = 16,
    num_layers = 5,
    heads = 8,
    dropout = 0.1,
    post_mlp_width_mult = 2.0,
    readout_method = 'mean'
)

print(model)
num_params = sum(p.numel() for p in model.parameters())
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {num_params:,}, Trainable: {num_trainable_params:,}")

# ------------------------
# Load data
# ------------------------

data_dir = '/home/yandex/MLWG2025/danielvolkov/datasets/VQA_mmg_BERT_BeiT_6fus_2tg/'
train_dl, val_dl = load_data_w_pe(data_dir, batch_size=32)

# ------------------------
# Training loop example
# ------------------------

for input_batch in train_dl:
    # input_batch is already GPU-ready and filtered (no ptr)
    res = model(**input_batch)
    print(f"{res=}, {res.shape=}")
    break