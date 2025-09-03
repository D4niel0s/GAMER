import torch, torch.nn.functional as F, numpy as np, math, json, wandb, os, time, sys

from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torch import optim
from tqdm import tqdm

from modules.utils import soft_cross_entropy,vqa_score_from_soft_targets
from modules.mmg_builder import build_multimodal_graph
from modules.graphGPS import GraphGPSNet

sys.path.append('..')
from data.VQA.dataset import VQAGraphsDataset


# -------------------------------- #
# === Config / Hyperparameters === #
# -------------------------------- #

# Paths
VQA_W_EMBED_PATH = ''
ANSWER_2_IDX_JSON_PATH = ''
checkpoint_dir = "./checkpoints"
resume_checkpoint = None  # path to resume

# Training parameters
num_epochs = 10
batch_size = 16

num_workers = 8
pin_memory = True
persistent_workers = True           # supported only in modern PyTorch

grad_accum_steps = 2                # 1 to disable
max_grad_norm = 1.0

val_interval_updates = 200          # validate every N optimizer updates
val_batches = 10_000                # Partial validation - None to do full validation
checkpoint_interval_updates = 1000  # checkpoint every N optimizer updates
log_every_n_updates = 10
save_best = True
use_amp = True                      # set False to disable mixed precision


# Data params (Graph construction, etc.)
num_fusion_nodes = 6
num_text_global_nodes = 2
self_loops_in_image_graph = False
add_lap_pe = True
lap_pe_dim = 16
graph_constructor = build_multimodal_graph


device = 'cuda' if torch.cuda.is_available() else 'cpu'




# ------------------------------- #
# === Dataset / DataLoader(s) === #
# ------------------------------- #

dataset = load_from_disk(VQA_W_EMBED_PATH)

with open(ANSWER_2_IDX_JSON_PATH, "r") as f:
    answer2idx = json.load(f)


lap_pe_transform = AddLaplacianEigenvectorPE(k=lap_pe_dim, attr_name="lap_pe", is_undirected=True)

collate_func = lambda x: VQAGraphsDataset.vqa_collate_fn(
    x,
    add_lap_pe = add_lap_pe,
    lap_pe_transform = lap_pe_transform
)

train_ds = VQAGraphsDataset(
    dataset['train'],
    answer2idx,
    graph_builder = graph_constructor,
    self_loops = self_loops_in_image_graph,
    fusion_num = num_fusion_nodes,
    text_global_num = num_text_global_nodes
)

valid_ds = VQAGraphsDataset(
    dataset['validation'],
    answer2idx,
    graph_builder = graph_constructor,
    self_loops = self_loops_in_image_graph,
    fusion_num = num_fusion_nodes,
    text_global_num = num_text_global_nodes
)


train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_func,
    num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
)

# Here no shuffle according to internet (dudes on multiple ancient forums) - either way id doesn't matter so why bother shuffling
valid_loader = DataLoader(
    valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_func,
    num_workers=max(1, num_workers//2), pin_memory=pin_memory, persistent_workers=persistent_workers
)





# ------------------------------------- #
# === Model / Optimizer / Scheduler === #
# ------------------------------------- #

model = GraphGPSNet(
	node_dim = 768 + 5,         # 768 is BERT/BeiT embeds, 5 extra feats are node type one-hots
	edge_dim = 6, 		        # 6 feats are edge type one-hots
	num_tasks = 3_000,	        # For VQA we need to predict from 3,000 classes (top 3k answers from train)
	hidden_dim = 768,	        # idk man just match BERT or smth seems legit
	pe_dim = lap_pe_dim,
	num_layers = 5,	            # Graph is connected with max 5 hop distance
	heads = 8,         	        # Random ass number that seems cool
	dropout = 0.1,		        # idk man wtf is dropout
	post_mlp_width_mult = 2.0,
	readout_method = 'mean'
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {num_params: ,}")

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {num_trainable_params: ,}")


optimizer = optim.Adam(model.parameters(), lr=1e-4)

total_updates = math.ceil(len(train_loader) * num_epochs / grad_accum_steps)
scheduler = CosineAnnealingLR(optimizer, T_max=total_updates)


scaler = GradScaler(device=device, enabled=use_amp)




# ----------------------------------------------------------- #
# === Load checkpoint (and opt + sched + amp state dicts) === #
# ----------------------------------------------------------- #

start_epoch = 0
global_update = 0
best_val_score = -float('inf')

if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
    ckpt = torch.load(resume_checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    scaler.load_state_dict(ckpt.get('scaler', {}))
    start_epoch = ckpt['epoch'] + 1
    global_update = ckpt['global_update']
    best_val_score = ckpt.get('best_val_score', best_val_score)
    print(f"Resumed from {resume_checkpoint} (epoch {start_epoch}, update {global_update})")



# helper to move entire dict to device
def move_batch_to_device(batch, device, non_blocking=False):
    return {k: (v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

# ----------------------------------------------------------- #
# === Train loop! (With wandb logging and all the frills) === #
# ----------------------------------------------------------- #

# === WandB init ===
wandb.init(project="GAMER", config=dict(
    dataset = "VQA v2",
    # Training parameters
    epochs = num_epochs,
    batch_size = batch_size,
    grad_accum_steps = grad_accum_steps,
    max_grad_norm = max_grad_norm,

    # Data params (Graph construction, etc.)
    num_fusion_nodes = num_fusion_nodes,
    num_text_global_nodes = num_text_global_nodes,
    self_loops_in_image_graph = self_loops_in_image_graph,
    add_lap_pe = add_lap_pe,
    lap_pe_dim = lap_pe_dim,
    graph_constructor = build_multimodal_graph,    
))



model.train()
optimizer.zero_grad()

try:
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}", leave=False)
        for step, (inputs, labels) in pbar:

            inputs = move_batch_to_device(inputs, device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

            with autocast(device_type='cuda', enabled=use_amp):
                logits = model(**inputs)                # [B, C]
                loss = soft_cross_entropy(logits, labels)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum_steps
            epoch_steps += 1

            # only step optimizer on accumulation boundary
            if (step + 1) % grad_accum_steps == 0:
                # unscale + clip + step
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_update += 1

                # logging
                if global_update % log_every_n_updates == 0:
                    train_vqa_acc = vqa_score_from_soft_targets(logits.detach(), labels.detach())
                    current_lr = scheduler.get_last_lr()[0]
                    wandb.log({
                        "train/loss": epoch_loss / epoch_steps,
                        "train/vqa_acc_batch": train_vqa_acc,
                        "train/grad_norm": grad_norm,
                        "lr": current_lr,
                        "global_update": global_update
                    }, step=global_update)

                # Validation (on optimizer updates)
                if global_update % val_interval_updates == 0:
                    model.eval()
                    val_loss = 0.0
                    val_steps = 0
                    val_vqa_acc = 0.0
                    with torch.no_grad():
                        for val_batch_num, (val_inputs, val_labels) in enumerate(valid_loader):
                            if val_batches is not None and val_batch_num >= val_batches:
                                break

                            val_inputs = move_batch_to_device(val_inputs, device, non_blocking=pin_memory)
                            val_labels = val_labels.to(device, non_blocking=pin_memory)

                            with autocast(device_type='cuda', enabled=use_amp):
                                val_logits = model(**val_inputs)
                                val_loss += soft_cross_entropy(val_logits, val_labels).item()
                                val_vqa_acc += vqa_score_from_soft_targets(val_logits, val_labels)
                                val_steps += 1
                    val_loss = val_loss / max(1, val_steps)
                    val_vqa_acc = val_vqa_acc / max(1, val_steps)
                    wandb.log({"val/loss": val_loss, "val/vqa_acc": val_vqa_acc}, step=global_update)
                    model.train()

                    # best checkpoint
                    if save_best and val_vqa_acc > best_val_score:
                        best_val_score = val_vqa_acc
                        best_ckpt = os.path.join(checkpoint_dir, f"best_ckpt_update_{global_update}.pt")
                        torch.save({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'scaler': scaler.state_dict() if use_amp else None,
                            'epoch': epoch,
                            'global_update': global_update,
                            'best_val_score': best_val_score
                        }, best_ckpt)
                        print(f"Saved new best checkpoint: {best_ckpt}")

                # periodic checkpoint
                if global_update % checkpoint_interval_updates == 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"ckpt_update_{global_update}.pt")
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict() if use_amp else None,
                        'epoch': epoch,
                        'global_update': global_update,
                        'best_val_score': best_val_score
                    }, ckpt_path)
                    print(f"Saved checkpoint at update {global_update}: {ckpt_path}")

            # progress bar update
            if (step + 1) % max(1, len(train_loader)//100) == 0:
                pbar.set_postfix({"loss": epoch_loss / epoch_steps if epoch_steps else 0.0})

        # if leftover gradients because dataloader ended not divisible by grad_accum_steps:
        if (step + 1) % grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_update += 1
            print("Flushed leftover gradients at epoch end (performed final optimizer.step())")

        # epoch summary
        epoch_time = time.time() - t0
        wandb.log({"epoch": epoch, "epoch/train_loss": epoch_loss / max(1, epoch_steps), "epoch_time": epoch_time}, step=global_update)
        print(f"Epoch {epoch} finished. avg loss {epoch_loss / max(1, epoch_steps):.4f} time {epoch_time:.1f}s")

except KeyboardInterrupt:
    print("KeyboardInterrupt caught — saving last checkpoint.")
    ckpt_path = os.path.join(checkpoint_dir, f"interrupt_ckpt_update_{global_update}.pt")
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict() if use_amp else None,
        'epoch': epoch,
        'global_update': global_update,
        'best_val_score': best_val_score
    }, ckpt_path)
    raise