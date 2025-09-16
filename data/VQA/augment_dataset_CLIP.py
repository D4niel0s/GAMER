import torch

from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load dataset and BERT
# -------------------------
dataset = load_dataset("pingzhili/vqa_v2")

model_name = "openai/clip-vit-base-patch16"
clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True).eval().to(device)


# =========================
# Batched embedding function
# =========================

@torch.inference_mode()
def encode_batch(questions, images):
    rgb_images = [img.convert("RGB") if isinstance(img, Image.Image) else img for img in images]

    inputs = clip_processor(text=questions, images=rgb_images, return_tensors="pt", padding=True).to(device)
    attn_mask = inputs['attention_mask']

    outputs = clip_model(**inputs, output_hidden_states=True)

    raw_patches = outputs.vision_model_output.last_hidden_state[:, 1:, :] # skip [CLS]
    raw_tokens   = outputs.text_model_output.last_hidden_state

    patch_embeds = clip_model.visual_projection(raw_patches)
    token_embeds = clip_model.text_projection(raw_tokens)

    txt_embeds = [e[attn_mask[i].bool()] for i,e in enumerate(token_embeds)]

    return txt_embeds, patch_embeds


# =========================
# Map function with batching
# =========================
def process_batch(batch):
    q_emb, i_emb = encode_batch(batch["question"], batch["image"])
    return {
        "question_embedding": q_emb,
        "image_embedding": [arr for arr in i_emb],
    }

# =========================
# Apply batched processing
# =========================
vqa_emb = dataset.map(
    process_batch,
    batched=True,
    batch_size=16,   # adjust for your VRAM
    remove_columns=[],
)

# Save to disk
vqa_emb.save_to_disk("vqa_v2_with_embeddings")
print("✅ Done! Dataset saved with CLIP embeddings.")