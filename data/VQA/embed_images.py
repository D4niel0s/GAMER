import torch
from transformers import BeitImageProcessor, BeitModel
from datasets import load_dataset, Dataset, Image as DSImage
from PIL import Image as PILImage
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load VQA dataset and BEiT
# -------------------------
dataset = load_dataset("pingzhili/vqa_v2")
dataset = dataset.cast_column("image", DSImage(decode=False))  # keep paths only

beit_processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
beit_model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224', use_safetensors=True).eval().to(device)

# -------------------------
# Deduplicate images across all splits using Pandas
# -------------------------
dfs = []
for split in dataset.keys():
    print('Deduplicating split:', split)
    df = dataset[split].to_pandas()
    df = df.drop_duplicates(subset="image_id")
    dfs.append(df)

# Concatenate splits and deduplicate globally
df_all = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="image_id")
print(f"Total unique images across all splits: {len(df_all)}")

# Convert back to HF Dataset
unique_img_ds = Dataset.from_pandas(df_all, preserve_index=False)

# -------------------------
# Embed images in batches
# -------------------------
def embed_images(batch):
    images = [PILImage.open(p).convert("RGB") for p in batch["image"]]
    inputs = beit_processor(images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = beit_model(**inputs)
    # drop CLS token, store patch embeddings
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
