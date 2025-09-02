import torch

from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from transformers import (
    BertTokenizer, BertModel,
    BeitImageProcessor, BeitModel
)


device = "cuda" if torch.cuda.is_available() else "cpu"
d_embed = 768 # Both BEiT-base and BERT-base have hidden size 768

# -------------------------
# Load dataset and BERT
# -------------------------
dataset = load_dataset("pingzhili/vqa_v2")

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").eval().cuda()

beit_processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
beit_model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224', use_safetensors=True).eval().to(device)

# =========================
# Batched embedding functions
# =========================
@torch.no_grad()
def encode_questions(questions):
    inputs = bert_tokenizer(questions, return_tensors="pt", padding=True).to(device)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.cpu().numpy()  # CLS token

@torch.no_grad()
def encode_images(images):
    inputs = beit_processor(images=images, return_tensors="pt").to(device)
    outputs = beit_model(**inputs, output_hidden_states=True)
    return outputs.last_hidden_state[:, 1:, :].cpu().numpy()  # contextualized patch embeddings


# =========================
# Map function with batching
# =========================
def process_batch(batch):
    # batch["question"] is a list of strings
    q_emb = encode_questions(batch["question"])
    i_emb = encode_images(batch["image"])
    return {
        "question_embedding": [arr for arr in q_emb],
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
print("✅ Done! Dataset saved with batched BERT & BEiT embeddings.")