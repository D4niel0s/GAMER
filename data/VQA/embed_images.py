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
unique_imgs = {}
for split in dataset.keys():
    for img, img_id in tqdm(zip(dataset[split]["image"], dataset[split]["image_id"]), desc=f"Deduplicating images in {split}", total=len(dataset[split])):
        if img_id not in unique_imgs:
            unique_imgs[img_id] = img
print(f"Unique images: {len(unique_imgs)}")


# -------------------------
# Compute BEiT embeddings once
# -------------------------
def embed_batch(images: list[Image.Image]):
    inputs = beit_processor(images, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = beit_model(**inputs)
    return outputs.last_hidden_state[:, 1:, :].cpu().numpy() # Throw CLS token


imgid2emb = {}
img_ids = list(unique_imgs.keys())
batch_size = 32

for i in tqdm(range(0, len(img_ids), batch_size), desc="Computing BEiT embeddings"):
    batch_ids = img_ids[i:i+batch_size]
    images = [unique_imgs[id].convert("RGB") for id in batch_ids]

    embeds = embed_batch(images)  # (B, P, D)

    for id, e in zip(batch_ids, embeds):
        imgid2emb[str(id)] = e   # keys must be str for npz


# -------------------------
# Save all image embeddings to NPZ
# -------------------------
np.savez_compressed("VQA_img_beit_embeds.npz", **imgid2emb)
print("Saved all BEiT embeddings!")