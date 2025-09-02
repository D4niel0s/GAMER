import torch, numpy as np

from transformers import BeitImageProcessor, BeitModel
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Load dataset and BEiT
# -------------------------
dataset = load_dataset("pingzhili/vqa_v2")

beit_processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
beit_model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224', use_safetensors=True).eval().cuda()


# -------------------------
# Deduplicate images
# -------------------------
# dedup with ids
unique_ids = set()
for split in dataset.keys():
    unique_ids.update(dataset[split]["image_id"])
print(f"Unique images: {len(unique_ids)}")


# Build mapping image_id → image (once per unique id)
id2img = {}
for split in dataset.keys():
    for row in tqdm(dataset[split], desc=f"Building id2img from {split}"):
        img_id = row["image_id"]
        if img_id not in id2img:
            id2img[img_id] = row["image"]   # PIL.Image
        if len(id2img) == len(unique_ids):
            break
print(f"Unique mapped ids: {len(id2img)}")


# -------------------------
# Compute BEiT embeddings once
# -------------------------
def embed_batch(images: list[Image.Image]):
    inputs = beit_processor(images, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = beit_model(**inputs)
    return outputs.last_hidden_state[:, 1:, :].cpu().numpy() # Throw CLS token


imgid2emb = {}
img_ids = list(id2img.keys())
batch_size = 32

for i in tqdm(range(0, len(img_ids), batch_size), desc="Computing BEiT embeddings"):
    batch_ids = img_ids[i:i+batch_size]
    images = [id2img[id].convert("RGB") for id in batch_ids]

    embeds = embed_batch(images)  # (B, P, D)

    for id, e in zip(batch_ids, embeds):
        imgid2emb[str(id)] = e   # keys must be str for npz


# -------------------------
# Save all image embeddings to NPZ
# -------------------------
np.savez_compressed("VQA_img_beit_embeds.npz", **imgid2emb)
print("Saved all BEiT embeddings!")