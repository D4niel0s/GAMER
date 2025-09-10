import torch,json, os, sys

from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch.amp.autocast_mode import autocast
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
from data.VQA.dataset import VQAGraphsDataset, VQAGraphsDataset_TEST


# helper to move entire dict to device
def move_batch_to_device(batch, device, non_blocking=False):
    return {k: (v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}



@torch.inference_mode()
def evaluate_val(model, loader, device, use_amp=True, max_batches=None):
    """Evaluate on validation split with VQA accuracy."""

    model.eval()
    total_acc, steps = 0.0, 0
    pbar = tqdm(enumerate(loader), total=max_batches if max_batches else len(loader), desc="Validation")
    for i, (inputs, labels) in pbar:
        if max_batches and i >= max_batches:
            break

        inputs = move_batch_to_device(inputs, device)
        labels = labels.to(device)

        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(**inputs)

        acc = vqa_score_from_soft_targets(logits, labels)
        total_acc += acc
        steps += 1
        pbar.set_postfix({"acc": total_acc / steps})

    return total_acc / max(1, steps)



@torch.inference_mode()
def predict_test(model, test_loader, idx2ans, device, use_amp=True, output_json="test_predictions.json"):
    """Run inference on test set and dump predictions in VQA format."""
    model.eval()
    results = []
    pbar = tqdm(test_loader, desc="Test inference")
    for inputs, qids in pbar:

        inputs = move_batch_to_device(inputs, device)
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(**inputs)

        # Get argmax prediction
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        answers = [idx2ans[pid] for pid in pred_ids]

        for qid, ans in zip(qids, answers):
            results.append({"question_id": int(qid), "answer": ans})

    with open(output_json, "w") as f:
        json.dump(results, f)
    print(f"Saved test predictions to {output_json}")
    return output_json



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
    batch_size = cfg['batch_size']

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


    valid_collate_func = lambda x: VQAGraphsDataset.vqa_collate_fn(
        x,
        add_lap_pe = add_lap_pe,
        lap_pe_transform = lap_pe_transform
    )
    valid_ds = VQAGraphsDataset(
        dataset['validation'],
        answer2idx,
        graph_builder = graph_constructor,
        self_loops = self_loops_in_image_graph,
        fusion_num = num_fusion_nodes,
        text_global_num = num_text_global_nodes
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, collate_fn=valid_collate_func,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
    )


    test_collate_func = lambda x: VQAGraphsDataset_TEST.vqa_test_collate_fn(
        x,
        add_lap_pe = add_lap_pe,
        lap_pe_transform = lap_pe_transform
    )
    test_ds = VQAGraphsDataset_TEST(
        dataset['test'],
        answer2idx,
        graph_builder = graph_constructor,
        self_loops = self_loops_in_image_graph,
        fusion_num = num_fusion_nodes,
        text_global_num = num_text_global_nodes
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=test_collate_func,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
    )

    

    # ------------- #
    # === Model === #
    # ------------- #

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
    
    print('Model:')
    print(model)


    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded {resume_checkpoint} to model")
    else:
        raise ValueError("Checkpoint path is bad.")

    
    # Run val evaluation
    val_acc = evaluate_val(model, valid_loader, device, use_amp=cfg["use_amp"], max_batches=None)
    print(f"Validation VQA accuracy: {val_acc:.4f}")

    # Run test prediction dump
    idx2ans = {v:k for k,v in answer2idx.items()}
    
    output_json = predict_test(model, test_loader, idx2ans, device, use_amp=cfg["use_amp"])
    print(f"Test predictions saved to {output_json}")


if __name__ == "__main__":
    main()