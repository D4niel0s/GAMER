import torch, numpy as np

from transformers import BertTokenizer, BertModel
from datasets import load_dataset, Array2D


device = "cuda" if torch.cuda.is_available() else "cpu"
d_embed = 768 # Both BEiT-base and BERT-base have hidden size 768

# -------------------------
# Load dataset and BERT
# -------------------------
dataset = load_dataset("pingzhili/vqa_v2")

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").eval().cuda()


# -------------------------
# Attach question + image embeddings to dataset
# -------------------------

# Reload dict from npz
data = np.load("VQA_img_beit_embeds.npz", allow_pickle=True)
imgid2emb = {k: data[k] for k in data.files}


new_features = dataset["train"].features.copy()
new_features["question_emb"] = Array2D(dtype="float32", shape=(None, d_embed))
new_features["image_emb"] = Array2D(dtype="float32", shape=(None, d_embed))



def embed_text(texts: list[str]):
    inputs = bert_tokenizer(texts, return_tensors="pt", padding=True).to('cuda')
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.cpu().numpy()

def process_batch(batch):
    questions = batch['question']
    image_ids = batch['image_id']

    # Text → all token embeddings
    text_embs = embed_text(questions)  # (B, T, D)

    # Image → lookup from dict
    img_embs = [imgid2emb[str(img_id)] for img_id in image_ids] 

    batch['question_emb'] = list(text_embs)
    batch['image_emb'] = img_embs
    return batch

dataset = dataset.map(
    process_batch,
    batched=True,
    batch_size=32,
    features=new_features,
    desc='Adding question + image embeddings'
)

dataset.save_to_disk('vqa_with_embeds_dedup')
print('Saved enriched dataset to vqa_with_embeds_dedup')
