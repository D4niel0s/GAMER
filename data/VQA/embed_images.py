import torch
from transformers import BeitImageProcessor, BeitModel
from datasets import load_dataset, concatenate_datasets, Dataset, Image as DSImage
from PIL import Image as PILImage
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load VQA dataset and BEiT
# -------------------------
dataset = load_dataset("pingzhili/vqa_v2")
dataset = dataset.cast_column("image", DSImage(decode=False))

beit_processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
beit_model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224', use_safetensors=True).eval().to(device)

# -------------------------
# Concatenate splits
# -------------------------
all_ds = concatenate_datasets([dataset[split] for split in dataset.keys()])
print(f"Total images before deduplication: {len(all_ds)}")

# -------------------------
# Deduplicate images using map + set
# -------------------------
seen = set()
def keep_first(example):
    img_id = example["image_id"]
    if img_id in seen:
        return None
    
    seen.add(img_id)
    return example

unique_img_ds = all_ds.map(keep_first, batched=False)
# Filter out None entries
unique_img_ds = unique_img_ds.filter(lambda x: x is not None)

print(f"Unique images across all splits: {len(unique_img_ds)}")

# -------------------------
# Embed images in batches
# -------------------------
def embed_images(batch):
    images = [PILImage.open(p).convert("RGB") for p in batch["image"]]
    inputs = beit_processor(images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = beit_model(**inputs)
    # Drop CLS token, store patch embeddings
    batch["image_emb"] = [emb.cpu().numpy() for emb in outputs.last_hidden_state[:, 1:, :]]
    return batch

batch_size = 32
unique_img_ds = unique_img_ds.map(
    embed_images,
    batched=True,
    batch_size=batch_size,
    remove_columns=["image"]
)

# -------------------------
# Save the dataset
# -------------------------
unique_img_ds.save_to_disk("VQA_img_beit_embeds_ds")
print("Saved all BEiT embeddings as HF dataset: VQA_img_beit_embeds_ds")