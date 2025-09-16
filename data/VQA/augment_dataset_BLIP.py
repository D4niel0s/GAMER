import torch

from transformers import BlipProcessor, BlipModel
from datasets import load_dataset
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load dataset and BERT
# -------------------------
dataset = load_dataset("pingzhili/vqa_v2")

model_name = "Salesforce/blip-vqa-base"

blip_processor = BlipProcessor.from_pretrained(model_name, size=dict(height=224, width=224))
blip_model = BlipModel.from_pretrained(model_name, use_safetensors=True).eval().to('cuda')


# =========================
# Batched embedding function
# =========================

@torch.inference_mode()
def encode_batch(questions, images):
    rgb_images = [img.convert("RGB") if isinstance(img, Image.Image) else img for img in images]

    inputs = blip_processor(text=questions, images=rgb_images, return_tensors="pt", padding=True).to(device)
    attn_mask = inputs['attention_mask']

    outputs = blip_model(**inputs, output_hidden_states=True)

    patch_embeds = outputs.vision_model_output.last_hidden_state[:, 1:, :] # skip [CLS]
    token_embeds = outputs.text_model_output.last_hidden_state

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
print("✅ Done! Dataset saved with BLIP embeddings.")