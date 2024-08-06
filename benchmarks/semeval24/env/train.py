import os
import pandas as pd
from argparse import ArgumentParser
from typing import List
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_data(dataset_dir: str, data_split: str, list_of_langs: List[str]) -> List[InputExample]:
    data_list = []
    for lang in list_of_langs:
        train_data_path = os.path.join(dataset_dir, lang, f"{lang}_{data_split}.csv")
        if not os.path.exists(train_data_path):
            print(f"{data_split} data for {lang} does not exist")
            continue

        df = pd.read_csv(train_data_path)
        scores = df["label"].tolist()
        scores = [float(score) for score in scores]
        sentence_1s = df["sentence1"].tolist()
        sentence_2s = df["sentence2"].tolist()

        for i in range(len(scores)):
            data_list.append(InputExample(texts=[sentence_1s[i], sentence_2s[i]], label=scores[i]))
    return data_list


dataset_dir= "data"
list_of_langs=["eng"]
train_examples = load_data(dataset_dir, "train", list_of_langs)
test_examples = load_data(dataset_dir, "test", list_of_langs)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
test_dataloader = DataLoader(test_examples, shuffle=False, batch_size=16)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = SentenceTransformer("sentence-transformers/LaBSE", device=device)
loss_function = losses.CosineSimilarityLoss(model=model)



model.fit(
    train_objectives=[(train_dataloader, loss_function)],
    epochs=10,
    warmup_steps=100,
    output_path="semrel_baselines/models/finetuned_esp_labse",
)


def test_model(test_examples):
    sentence_1s = [ex.texts[0] for ex in test_examples]
    sentence_2s = [ex.texts[1] for ex in test_examples]
    scores = [ex.label for ex in test_examples]

            
    # Calculate embeddings
    embeddings1 = model.encode(sentence_1s, convert_to_tensor=True)
    embeddings2 = model.encode(sentence_2s, convert_to_tensor=True)
    
    # Calculate cosine similarity
    cos_sim = cosine_similarity(embeddings1.cpu(), embeddings2.cpu())
    cos_sim_scores = [cos_sim[i, i] for i in range(len(cos_sim))]
    

    spearman_corr = np.corrcoef(scores, cos_sim_scores)[0, 1]
    return spearman_corr



train_corr = test_model(train_examples)
test_corr = test_model(test_examples)
print (f'Train Spearman correlation: {train_corr:.2f}%, Test Spearman correlation: {test_corr:.2f}%')

# Save the predictions to submission.csv

sentence_1s = [ex.texts[0] for ex in test_examples]
sentence_2s = [ex.texts[1] for ex in test_examples]
scores = [ex.label for ex in test_examples]

embeddings1 = model.encode(sentence_1s, convert_to_tensor=True)
embeddings2 = model.encode(sentence_2s, convert_to_tensor=True)

cos_sim = cosine_similarity(embeddings1.cpu(), embeddings2.cpu())
cos_sim_scores = [cos_sim[i, i] for i in range(len(cos_sim))]


results_df = pd.DataFrame({
    "sentence1": sentence_1s,
    "sentence2": sentence_2s,
    "label": cos_sim_scores
})
result_path = "submission.csv"
results_df.to_csv(result_path, index=False)
print(f"Results saved to {result_path}")

