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




import warnings, torch, requests
warnings.filterwarnings("ignore")
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from utils import preprocess_dataset, batch_to_model_inputs # graph -> batch + PE
from graphGPS import GraphGPSNet # model
from datasets import load_from_disk

def row_to_data(row):
    x = torch.tensor(row["x"], dtype=torch.float)
    edge_index = torch.tensor(row["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(row["edge_attr"], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def make_transform(k=16, is_undirected=True):
    return AddLaplacianEigenvectorPE(k=k, attr_name="lap_pe", is_undirected=is_undirected)

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
        return data

def load_data_w_pe(data_dir, splits=['train', 'validation'], batch_size=32, shuffle=True):
    transform = make_transform(k=16)
    hf_dataset = load_from_disk(data_dir)
    loaders = []
    for split in splits:
        dataset = HFDataset(hf_dataset[split], column="multimodal_graph", transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        loaders.append(loader)
    return tuple(loaders)

# Setup
data_dir = '/home/yandex/MLWG2025/danielvolkov/datasets/VQA_mmg_BERT_BeiT_6fus_2tg/'
model = GraphGPSNet(
    node_dim = 768 + 5,
    edge_dim = 6,
    num_tasks = 3_000,
    hidden_dim = 768,
    pe_dim = 16,
    num_layers = 5,
    heads = 8,
    dropout = 0.1,
    post_mlp_width_mult = 2.0,
    readout_method = 'mean'
).cuda()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# DEBUGGING: Let's isolate the problem step by step

print("\n=== STEP 1: Test single batch without loop ===")
train_dl, val_dl = load_data_w_pe(data_dir, batch_size=32)
batch = next(iter(train_dl))
print(f"Batch type: {type(batch)}")
print(f"Batch keys: {batch.keys if hasattr(batch, 'keys') else 'No keys attr'}")

print("\n=== STEP 2: Test batch_to_model_inputs ===")
input_batch = batch_to_model_inputs(batch)
print(f"Input batch type: {type(input_batch)}")
print(f"Input batch keys: {list(input_batch.keys()) if hasattr(input_batch, 'keys') else 'No keys'}")

# Check what's actually in input_batch
for k, v in input_batch.items():
    if torch.is_tensor(v):
        print(f"  {k}: {v.shape} ({v.dtype}) - {v.numel()} elements")
    else:
        print(f"  {k}: {type(v)} - {v}")

print("\n=== STEP 3: Test GPU transfer ===")
gpu_batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in input_batch.items()}
print("GPU transfer successful")

print(f"GPU memory after transfer: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

print("\n=== STEP 4: Test model forward ===")
model.train()
res = model(**gpu_batch)
print(f"Forward pass successful, output shape: {res.shape}")
print(f"GPU memory after forward: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# Now let's test the actual loop to see where it breaks
print("\n=== STEP 5: Test loop (this is where it probably breaks) ===")

# SUSPECTED ISSUES TO CHECK:
# 1. Are you accumulating gradients without clearing them?
# 2. Is batch_to_model_inputs creating references that don't get cleaned up?
# 3. Is the DataLoader holding onto batches?
# 4. Is AddLaplacianEigenvectorPE creating persistent references?

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i, batch in enumerate(train_dl):
    if i >= 5:  # Just test first few batches
        break
    
    print(f"\n--- Batch {i} ---")
    print(f"Memory at start: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    # Standard training loop - NO manual memory management
    optimizer.zero_grad()
    
    input_batch = batch_to_model_inputs(batch)
    input_batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in input_batch.items()}
    
    print(f"Memory after GPU transfer: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    res = model(**input_batch)
    
    print(f"Memory after forward: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    print(f"Forward successful, output: {res.shape}")
    
    # Simulate loss and backward (comment out if this causes issues)
    # fake_loss = res.sum()  # Dummy loss
    # fake_loss.backward()
    # optimizer.step()
    
    print(f"Memory at end of iteration: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

print("\n=== DEBUGGING COMPLETE ===")

# COMMON ISSUES TO CHECK:
print("""
LIKELY CULPRITS:
1. Check your batch_to_model_inputs() function - is it creating copies?
2. Check if AddLaplacianEigenvectorPE is holding references
3. Are your graphs extremely large?
4. Is there a gradient accumulation bug?

Try these fixes:
- Reduce batch_size to 1 temporarily 
- Remove the PE transform temporarily
- Check the actual size of your graphs (num_nodes, num_edges)
- Look inside batch_to_model_inputs() for memory leaks
""")