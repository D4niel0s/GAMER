import torch, torch.nn as nn, math, json, wandb, os, time, sys

from torch_geometric.transforms import AddLaplacianEigenvectorPE
from transformers import get_cosine_schedule_with_warmup
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from datasets import load_from_disk
from pprint import pprint
from tqdm import tqdm

from modules.mmg_builder import build_multimodal_graph
from modules.cayley_builder import build_cayley_graph
from modules.utils import vqa_score_from_soft_targets
from config import get_parser, get_model_config
from modules.graphGPS import GraphGPSNet

sys.path.append('..')
from data.VQA.dataset import VQAGraphsDataset


# helper to move entire dict to device
def move_batch_to_device(batch, device, non_blocking=False):
    return {k: (v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}



# -------------------------------
# Validation function
# -------------------------------
@torch.no_grad()
def validate(model, val_loader, criterion, device, move_fn, val_batches=None, use_amp=True):
    model.eval()
    val_loss = 0.0
    val_steps = 0
    val_vqa_acc = 0.0

    for val_batch_num, (val_inputs, val_labels) in enumerate(val_loader):
        if val_batches is not None and val_batch_num >= val_batches:
            break

        val_inputs = move_fn(val_inputs, device)
        val_labels = val_labels.to(device)

        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            val_logits = model(**val_inputs)
            val_loss += criterion(val_logits, val_labels).item()
            val_vqa_acc += vqa_score_from_soft_targets(val_logits, val_labels)
            val_steps += 1

    val_loss /= max(1, val_steps)
    val_vqa_acc /= max(1, val_steps)
    model.train()
    return val_loss, val_vqa_acc




# -------------------------------
# Main training function
# -------------------------------
def main():

    # -------------------------------- #
    # === Config / Hyperparameters === #
    # -------------------------------- #
    cfg = vars(get_parser().parse_args())
    print('Config:')
    pprint(cfg)

    # Paths
    VQA_W_EMBED_PATH = cfg['dataset_path']
    ANSWER_2_IDX_JSON_PATH = cfg['ans2idx_path']
    checkpoint_dir = cfg['checkpoint_dir']
    resume_checkpoint = cfg['resume_checkpoint']

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training params / options
    num_epochs = cfg['num_epochs']
    batch_size = cfg['batch_size']
    grad_accum_steps = cfg['grad_acc_steps']
    max_grad_norm = cfg['max_grad_norm']

    learning_rate = cfg['adamw_lr']
    weight_decay = cfg['adamw_weight_decay']
    warmup_fraction = cfg['warmup_fraction']

    val_interval_updates = cfg['val_interval_updates']
    val_batches = cfg['val_batches']
    checkpoint_interval_updates = cfg['checkpoint_interval_updates']
    log_every_n_updates = cfg['log_every_n_updates']
    save_best = cfg['save_best']
    use_amp = cfg['use_amp']

    # Graph construction options
    add_lap_pe = cfg['add_lap_pe']
    lap_pe_dim = cfg['lap_pe_dim'] if add_lap_pe else 0

    match cfg['graph_construction_method']:
        case 'mmg':
            graph_constructor = build_multimodal_graph
            model_node_dim = 768 + 5    # 768 is BERT / BEiT embeds. +5 feats are one-hot type feats concatted to embeds.
            model_edge_dim = 6          # one-hot type feats for edges
        case 'cayley':
            graph_constructor = build_cayley_graph
            model_node_dim = 768        # 768 is BERT / BEiT embeds. No additional feats since all nodes are the same here.
            model_edge_dim = 0          # No edge feats since all edges are the same 'type'.
        case _: # default
            raise ValueError('Graph construction method must be mmg or cayley!')
        
    num_fusion_nodes = cfg['num_fusion_nodes']
    num_text_global_nodes = cfg['num_text_global_nodes']
    self_loops_in_image_graph = cfg['self_loops_in_image_graph']

    # DataLoader and moving to GPU optimizations
    num_workers = cfg['num_workers']
    pin_memory = cfg['pin_memory']
    persistent_workers = cfg['persistent_workers']

    # Should be cuda but here to prevent program from crashing 💀
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

    # Traditionally no shuffle, but we do partial validation so to preserve I.I.D sampling for validation we need to shuffle
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_func,
        num_workers=max(1, num_workers//2), pin_memory=pin_memory, persistent_workers=persistent_workers
    )





    # ------------------------------------------------- #
    # === Model / Optimizer / Scheduler / Criterion === #
    # ------------------------------------------------- #

    model_conf = dict(
        node_dim = model_node_dim,
        edge_dim = model_edge_dim,
        output_dim = len(answer2idx),
        pe_dim = lap_pe_dim,
        **get_model_config()
    )
    model = GraphGPSNet(**model_conf).to(device)

    print('Model config:')
    pprint(model_conf)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params: ,}")

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {num_trainable_params: ,}")


    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_updates = math.ceil(len(train_loader) * num_epochs / grad_accum_steps)
    warmup_steps = int(warmup_fraction * total_updates)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates
    )

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=use_amp, device=device)

    # ----------------------------------------------------------- #
    # === Load checkpoint (and opt + sched state dicts) === #
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



    # ----------------------------------------------------------- #
    # === Train loop! (With wandb logging and all the frills) === #
    # ----------------------------------------------------------- #

    # === WandB init ===
    wandb.init(project="GAMER", config=dict(
        dataset = "VQA v2",
        model_name = "GAMER GraphGPS",
        model_config = model_conf,
        **cfg
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

                # Forward + loss
                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(**inputs)
                    loss = criterion(logits, labels)
                    loss = loss / grad_accum_steps

                # Backward + grad accumulation
                scaler.scale(loss).backward()
                epoch_loss += loss.item() * grad_accum_steps
                epoch_steps += 1

                # Step optimizer
                if (step + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_update += 1

                    # Logging
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

                    # Validation
                    if global_update % val_interval_updates == 0:
                        val_loss, val_vqa_acc = validate(model, valid_loader, criterion, device,
                                                         move_batch_to_device, val_batches, use_amp)
                        
                        wandb.log({"val/loss": val_loss, "val/vqa_acc": val_vqa_acc}, step=global_update)

                        if save_best and val_vqa_acc > best_val_score:
                            best_val_score = val_vqa_acc
                            best_ckpt = os.path.join(checkpoint_dir, f"best_ckpt_update_{global_update}.pt")
                            torch.save({
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'scaler': scaler.state_dict(),
                                'epoch': epoch,
                                'global_update': global_update,
                                'best_val_score': best_val_score
                            }, best_ckpt)
                            print(f"Saved new best checkpoint: {best_ckpt}")

                    # Periodic checkpoint
                    if global_update % checkpoint_interval_updates == 0:
                        ckpt_path = os.path.join(checkpoint_dir, f"ckpt_update_{global_update}.pt")
                        torch.save({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'scaler': scaler.state_dict(),
                            'epoch': epoch,
                            'global_update': global_update,
                            'best_val_score': best_val_score
                        }, ckpt_path)
                        print(f"Saved checkpoint at update {global_update}: {ckpt_path}")

                # Progress bar
                if (step + 1) % max(1, len(train_loader)//100) == 0:
                    pbar.set_postfix({"loss": epoch_loss / epoch_steps})

            # Flush leftover gradients
            if (step + 1) % grad_accum_steps != 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_update += 1
                print("Flushed leftover gradients at epoch end.")

            # Epoch summary
            epoch_time = time.time() - t0
            wandb.log({"epoch": epoch, "epoch/train_loss": epoch_loss / epoch_steps, "epoch_time": epoch_time},
                      step=global_update)
            print(f"Epoch {epoch} finished. avg loss {epoch_loss / epoch_steps:.4f} time {epoch_time:.1f}s")

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught — saving last checkpoint.")
        ckpt_path = os.path.join(checkpoint_dir, f"interrupt_ckpt_update_{global_update}.pt")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'global_update': global_update,
            'best_val_score': best_val_score
        }, ckpt_path)
        raise


if __name__ == '__main__':
    main()