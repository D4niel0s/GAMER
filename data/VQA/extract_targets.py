import json

from collections import Counter
from datasets import load_dataset


dataset = load_dataset("pingzhili/vqa_v2")

# Count all answers in training set
answer_counter = Counter()
for item in dataset['train']:
    for ans_dict in item['answers']:
        answer_counter[ans_dict['answer']] += 1

# Take top 3000
top_answers = [ans for ans, _ in answer_counter.most_common(3000)]
answer2idx = {ans: idx for idx, ans in enumerate(top_answers)}
num_classes = len(top_answers)  # 3000

# Save to JSON file
with open('answer2idx.json', 'w') as f:
    json.dump(answer2idx, f, indent=4)
