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
import gc

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

# Assume dataset has column 'multimodal_graph' which contains dict with x,edge_index,edge_attrs
data_dir = '/home/yandex/MLWG2025/danielvolkov/datasets/VQA_mmg_BERT_BeiT_6fus_2tg/'
model = GraphGPSNet(
    node_dim = 768 + 5, # 768 is BERT/BeiT embeds, 5 extra feats are node type one-hots
    edge_dim = 6, # 6 feats are edge type one-hots
    num_tasks = 3_000, # For VQA we need to predict from 3,000 classes (answers)
    hidden_dim = 768, # idk man just match BERT or smth seems legit
    pe_dim = 16, # standard (I thinks?)
    num_layers = 5, # Graph is connected with max 5 hop distance
    heads = 8, # Random ass number that seems cool
    dropout = 0.1, # idk man wtf is dropout
    post_mlp_width_mult = 2.0,
    readout_method = 'mean'
).cuda()  # Move model to GPU once

print(model)
num_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {num_params: ,}")
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {num_trainable_params: ,}")

train_dl, val_dl = load_data_w_pe(data_dir)

# TRAINING LOOP with proper memory management
model.train()  # Enable training mode
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i, batch in enumerate(train_dl):
    if i >= 100:
        break
        
    try:
        # Zero gradients
        optimizer.zero_grad()
        
        # Move to GPU
        input_batch = batch_to_model_inputs(batch)
        input_batch = {k: v.cuda() for k, v in input_batch.items() if torch.is_tensor(v)}
        
        # Forward pass
        res = model(**input_batch)
        
        # Compute loss (you'll need to add your actual loss computation here)
        # loss = your_loss_function(res, targets)
        print(f"Batch {i}: Output shape: {res.shape}")
        
        # Backward pass (uncomment when you have loss)
        # loss.backward()
        # optimizer.step()
        
        # CRITICAL: Clean up GPU memory after each batch
        del res, input_batch, batch
        torch.cuda.empty_cache()
        
        # Force garbage collection every 10 iterations
        if i % 10 == 0:
            gc.collect()
            print(f"Completed {i} batches, GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
                
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"OOM at batch {i}, cleaning up...")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        else:
            raise e

print("Training loop skeleton completed successfully!")

# ALTERNATIVE: Gradient checkpointing for even more memory savings
"""
# If you still get OOM, use gradient checkpointing to trade compute for memory
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i, batch in enumerate(train_dl):
    if i >= 100:
        break
        
    optimizer.zero_grad()
    
    input_batch = batch_to_model_inputs(batch)
    input_batch = {k: v.cuda() for k, v in input_batch.items() if torch.is_tensor(v)}
    
    # Use gradient checkpointing to save memory during forward pass
    res = torch.utils.checkpoint.checkpoint(model, **input_batch)
    
    # loss = your_loss_function(res, targets)
    # loss.backward()
    # optimizer.step()
    
    del res, input_batch, batch
    torch.cuda.empty_cache()
"""

# ALTERNATIVE SOLUTION 3: Process smaller chunks
"""
# If you still get OOM, try processing in smaller chunks
def process_in_chunks(batch, model, chunk_size=4):
    # This assumes your batch can be split - you might need to adapt based on your batch structure
    results = []
    total_samples = batch.x.size(0) if hasattr(batch, 'x') else len(batch)
    
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        # Create mini-batch (this is pseudo-code, adapt to your batch structure)
        mini_batch = create_mini_batch(batch, start_idx, end_idx)
        
        with torch.no_grad():
            mini_result = model(**mini_batch)
            results.append(mini_result.cpu())  # Move to CPU immediately
            
        del mini_batch, mini_result
        torch.cuda.empty_cache()
    
    return torch.cat(results, dim=0)
"""