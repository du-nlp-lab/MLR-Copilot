import openai

api_key = 'sk-proj-ByZgmgOzCpRPuGXXiLLeT3BlbkFJ8EP0WoTbx4pBfI40bAce'
openai.api_key = api_key

def call_gpt4_api(user_message):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            # {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )
    
    # Extract the assistant's response
    assistant_message = response['choices'][0]['message']['content']
    
    # Print the assistant's response
    print("Assistant:", assistant_message)
    
    return assistant_message

# Example usage
# system_message = """You are an AI assistant whose primary goal is to propose innovative, rigorous, and valid methodologies to solve newly identified scientific problems derived from existing scientific literature, in order to empower researchers to pioneer groundbreaking solutions that catalyze breakthroughs in their fields."""
user_message = """Proposed Method
Method:

Dataset Augmentation and Preparation:

Multilingual Data Collection: Augment existing datasets with additional sentence pairs from diverse sources, including news articles, social media, and parallel corpora, specifically targeting low-resource languages.
Back-Translation: Use back-translation techniques to generate paraphrased sentence pairs, enhancing the dataset and ensuring it captures a wide range of semantic nuances.
Transformer-Based Embeddings:

Pretrained Multilingual Models: Utilize pretrained multilingual models like mBERT, XLM-R, and LASER to generate high-quality embeddings for each sentence pair.
Contextual Embeddings: Incorporate models like T5 and mT5 to generate contextual embeddings, capturing broader commonalities like topic, viewpoint, and temporal context.
Semantic Alignment with Cross-Lingual Models:

Aligning Embeddings: Use cross-lingual alignment techniques such as VecMap or MUSE to align embeddings across different languages, ensuring consistent semantic representation.
Contrastive Learning: Implement contrastive learning methods to train the model on distinguishing between similar and dissimilar sentence pairs, improving its ability to discern fine-grained semantic differences.
Regression and Ranking Model:

Ensemble Regression Models: Combine various regression models (e.g., linear regression, SVR, and neural network-based regression) to predict the semantic relatedness score for each sentence pair.
Ranking Mechanism: Implement a ranking mechanism using models like RankNet or LambdaMART to order sentence pairs based on their predicted semantic relatedness scores.
Evaluation and Fine-Tuning:

Spearman Rank Correlation: Use the Spearman rank correlation coefficient as the primary evaluation metric to assess the model’s performance in ranking sentence pairs.
Hyperparameter Optimization: Employ grid search or Bayesian optimization to fine-tune model hyperparameters, ensuring optimal performance.
Error Analysis and Iteration: Conduct a thorough error analysis to identify and address any systematic biases or shortcomings in the model, iterating on the approach as necessary.
Rationale:
By combining these innovative techniques, the proposed method aims to effectively address the task of predicting semantic textual relatedness across different languages, particularly focusing on low-resource languages. This approach leverages recent advancements in Transformer-based architectures and cross-lingual alignment techniques, ensuring a rigorous, valid, and generalizable solution.

Experiment Design
Experiment:

Objective:
Validate the proposed method for predicting Semantic Textual Relatedness (STR) across multiple languages, focusing on low-resource languages.

Dataset Preparation:

Data Sources: Collect sentence pairs from diverse sources such as news articles, social media, and parallel corpora, particularly for the targeted low-resource languages (e.g., Algerian Arabic, Amharic, Hausa).
Back-Translation: Implement back-translation techniques to generate paraphrased sentence pairs, enhancing the diversity and semantic richness of the dataset.
Embedding Generation:

Multilingual Embeddings: Generate embeddings for each sentence pair using pretrained multilingual models like mBERT, XLM-R, and LASER.
Contextual Embeddings: Use models like T5 and mT5 to generate contextual embeddings, capturing broader semantic relationships.
Model Training:

Semantic Alignment: Apply cross-lingual alignment techniques (VecMap, MUSE) to ensure consistent semantic representation across languages.
Contrastive Learning: Train the model using contrastive learning methods to enhance its ability to distinguish between similar and dissimilar sentence pairs.
Ensemble Regression: Train an ensemble of regression models (linear regression, SVR, neural networks) to predict the semantic relatedness scores.
Ranking Model: Implement a ranking model (RankNet, LambdaMART) to order sentence pairs based on their predicted relatedness scores.
Evaluation:

Evaluation Metric: Use the Spearman rank correlation coefficient to assess the model’s performance in ranking sentence pairs.
Cross-Validation: Perform k-fold cross-validation to ensure robustness and generalizability of the model.
Hyperparameter Tuning:

Optimization Techniques: Use grid search or Bayesian optimization to fine-tune model hyperparameters.
Iterative Refinement: Conduct iterative refinement based on error analysis to address any biases or shortcomings in the model.
Reproducibility:

Code and Data Sharing: Make the code and dataset publicly available to ensure reproducibility.
Documentation: Provide detailed documentation of the experimental setup, data preprocessing steps, and model training process.
Rationale:

Objective:
Clearly defining the objective ensures the experiment remains focused on validating the proposed method for predicting STR across multiple languages.

Dataset Preparation:
Collecting diverse sentence pairs and using back-translation enhances the dataset's semantic richness, making the experiment more comprehensive and robust.

Embedding Generation:
Using pretrained multilingual models and contextual embeddings captures a wide range of semantic relationships, essential for accurately predicting STR.

Model Training:
Applying semantic alignment and contrastive learning techniques improves the model's ability to handle multilingual data. An ensemble of regression models and a ranking mechanism ensures robust performance.

Evaluation:
Using the Spearman rank correlation coefficient and cross-validation provides a clear and reliable measure of the model’s effectiveness in ranking sentence pairs.

Hyperparameter Tuning:
Optimization techniques and iterative refinement ensure the model is fine-tuned for optimal performance, addressing any biases or shortcomings.

Reproducibility:
Sharing code and data, along with detailed documentation, ensures the experiment can be reproduced and validated by other researchers, enhancing its impact and credibilitytrain.py:import os
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

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_score(submission_folder = "../env"):
    submission_path = os.path.join(submission_folder, "submission.csv")
    submission = pd.read_csv(submission_path, index_col=0)
    preds = submission["label"].tolist()
    preds = [float(pred) for pred in preds] 
    lang = "eng"

    test_data_path = os.path.join(submission_folder, "data", lang, f"{lang}_test.csv")
    df = pd.read_csv(test_data_path)
    scores = df["label"].tolist()
    scores = [float(score) for score in scores]    

    spearman_corr = np.corrcoef(scores, preds)[0, 1]
    return spearman_corr
eval.py
if __name__ == "__main__":
    print(get_score())

prepare.py
from datasets import load_dataset
import shutil
import os

# Define the directory where the dataset should be saved
lan = "eng"
download_dir = os.path.join("../env/data", lan)

# Create the directory if it does not exist, or remove it if it already exists
if os.path.exists(download_dir):
    shutil.rmtree(download_dir)
os.makedirs(download_dir)

# Load the dataset with the specified splits
dataset = load_dataset("SemRel/SemRel2024", lan, split=['train', 'dev', 'test'])

# Save each split to a CSV file in the specified directory
for split_name, split_data in zip(['train', 'dev', 'test'], dataset):
    split_data.to_csv(f"{download_dir}/{lan}_{split_name}.csv", index=False)

# Print a message indicating where the dataset has been saved and display the splits
print(f"Dataset downloaded and saved to {download_dir}")
for split_name in ['train', 'dev', 'test']:
    print(f"{lan}_{split_name} split saved to {download_dir}/{lan}_{split_name}.csv")
You are a helpful research assistant. You have access to the following tools:
- List Files:
        Use this to navigate the file system.
        Usage:
        
Action: List Files
        Action Input: {
            "dir_path": [a valid relative path to a directory, such as "." or "folder1/folder2"]
        }
        Observation: [The observation will be a list of files and folders in dir_path or current directory is dir_path is empty, or an error message if dir_path is invalid.]


- Copy File:
        Use this to copy a file to a new location with a new name.
        Usage:
        
Action: Copy File
        Action Input: {
            "source": [a valid file name with relative path to current directory if needed],
            "destination": [a valid file name with relative path to current directory if needed]
        }
        Observation: [A success message if the file is copied successfully, or an error message if the file cannot be copied.]


- Undo Edit Script:
        Use this to undo the last edit of the python script.
        Usage:
        
Action: Undo Edit Script
        Action Input: {
            "script_name": [a valid python script name with relative path to current directory if needed]
        }
        Observation: [The observation will be the content of the script before the last edit. If the script does not exist, the observation will be an error message.]


- Execute Script:
        Use this to execute the python script. The script must already exist.
        Usage:
        
Action: Execute Script
        Action Input: {
            "script_name": [a valid python script name with relative path to current directory if needed]
        }
        Observation: [The observation will be output of the script or errors.]


- Request Help:
        Use this to request help from human. Use this only when the provided tools and files are not enough for accomplishing necessary steps, such as requesting API reference or installing a library. So you should check through the provided tools and files first.
        Usage:
        
Action: Request Help
        Action Input: {
            "request": [a detailed description on what to do]
        }
        Observation: [The observation will be the response from human.]


- Final Answer:
        Use this to provide the final answer to the current task.
        Usage:
        
Action: Final Answer
        Action Input: {
            "final_answer": [a detailed description on the final answer]
        }
        Observation: [The observation will be empty.]


- Understand File:
        Use this to read the whole file and understand certain aspects. You should provide detailed description on what to look for and what should be returned. To get a better understanding of the file, you can use Inspect Script Lines action to inspect specific part of the file.
        Usage:
        
Action: Understand File
        Action Input: {
            "file_name": [a valid file name with relative path to current directory if needed],
            "things_to_look_for": [a detailed description on what to look for and what should returned]
        }
        Observation: [The observation will be a description of relevant content and lines in the file. If the file does not exist, the observation will be an error message.]


- Inspect Script Lines:
        Use this to inspect specific part of a python script precisely, or the full content of a short script. The number of lines to display is limited to 100 lines. This is especially helpful when debugging.
        Usage:
        
Action: Inspect Script Lines
        Action Input: {
            "script_name": [a valid python script name with relative path to current directory if needed],
            "start_line_number": [a valid line number],
            "end_line_number": [a valid line number]
        }
        Observation: [The observation will be the content of the script between start_line_number and end_line_number . If the script does not exist, the observation will be an error message.]


- Edit Script (AI):
        Use this to do a relatively large but cohesive edit over a python script. Instead of editing the script directly, you should describe the edit instruction so that another AI can help you do this.
        Usage:
        
Action: Edit Script (AI)
        Action Input: {
            "script_name": [a valid python script name with relative path to current directory if needed. An empty sctipt will be created if it does not exist.],
            "edit_instruction": [a detailed step by step description on how to edit it.],
            "save_name": [a valid file name with relative path to current directory if needed]
        }
        Observation: [The observation will be the edited content of the script. If the script does not exist, the observation will be an error message. You should always double check whether the edit is correct. If it is far from correct, you can use the Undo Edit Script action to undo the edit.]


- Reflection:
        Use this to look over all the past steps and reflect. You should provide detailed description on what to reflect on and what should be returned.
        Usage:
        
Action: Reflection
        Action Input: {
            "things_to_reflect_on": [a detailed description on what to reflect on and what should be returned]
        }
        Observation: [The observation will be a the reflection.]


- Retrieve Dataset:
        Retrieve a suitable dataset based on a detailed description of the requirements. You can load the dataset later from save_dir using the load_from_disk function of the HuggingFace datasets library.
        Usage:
        
Action: Retrieve Dataset
        Action Input: {
            "instruction": [an instruction on how to generate the output from the input],
            "save_dir": [directory to save the generated dataset dict to. We recommend saving to data/retrieved/]
        }
        Observation: [The observation will be a success message if the dataset was retrieved successfully. Otherwise, an error message will be returned.]


- Retrieve Model:
        Retrieve a suitable model based on a detailed description of the requirements. You can obtain the model given the name using the transformers.AutoModelForSeq2SeqLM.from_pretrained function.
        Usage:
        
Action: Retrieve Model
        Action Input: {
            "instruction": [an instruction on how to generate the output from the input]
        }
        Observation: [The observation will be a list of suitable models. You can choose one of them based on the requirements.]


- Process Dataset:
        Process dataset based on a detailed description of the requirements. You can load the processed data later from save_dirs using the load_from_disk function of the HuggingFace datasets library. The input text will be in the model_input column and the output text will be in the model_output column.
        Usage:
        
Action: Process Dataset
        Action Input: {
            "instruction": [an instruction on how to generate the output from the input],
            "load_dirs": [directories to load the dataset dicts from, separated by colons],
            "save_dirs": [directories to save the processed dataset dicts to, separated by colons. The order should match the order of the loaded datasets. We recommend saving to data/processed/]
        }
        Observation: [The observation will be a success message if the data was processed successfully. Otherwise, an error message will be returned.]


- Train Model:
        Train a Seq2Seq model from HuggingFace transformers library using the processed datasets and given hyperparameters.
        Usage:
        
Action: Train Model
        Action Input: {
            "model_name": [name of the model to train],
            "load_dirs": [directories to load the dataset dicts from, separated by colons],
            "result_dir": [directory to save the trained model and tokenizer to. We recommend using results/{trial_id}/. The trained model will be available as {result_dir}/trained_model/ and the tokenizer will be available as {result_dir}/trained_tokenizer/.],
            "epochs": [number of epochs to train the model for],
            "batch_size": [batch size for training the model],
            "warmup_steps": [number of warmup steps for the optimizer],
            "weight_decay": [weight decay for the optimizer],
            "learning_rate": [learning rate for the optimizer]
        }
        Observation: [The observation will be a success message if the model was trained successfully. Otherwise, an error message will be returned.]


- Execute Model on Test Set:
        Execute a trained model on the test sets of specified dataset dicts.
        Usage:
        
Action: Execute Model on Test Set
        Action Input: {
            "result_dir": [directory where the trained model and tokenizer are saved],
            "load_dirs": [directories to load the dataset dicts from, separated by colons],
            "save_path": [file to save the results of the model execution in json format],
            "batch_size": [batch size for executing the model],
            "input_column": [column name of the input text]
        }
        Observation: [The observation will be a success message if the model was executed successfully. Otherwise, an error message will be returned.]


- Evaluate Model:
        Evaluate a trained model on the test sets of specified dataset dicts.
        Usage:
        
Action: Evaluate Model
        Action Input: {
            "load_dirs": [directories to load the dataset dicts from, separated by colons],
            "save_path": [file to load the results of the model execution in json format],
            "output_column": [column name of the output text]
        }
        Observation: [The values for various evaluation metrics will be returned.]




Research Problem: Instruction: The Task:
The paper addresses the task of automatically discerning the level of semantic relatedness between pairs of sentences. Specifically, the task involves predicting the Semantic Textual Relatedness (STR) of sentence pairs and ranking them based on their semantic proximity across 14 different languages.

The Drawback of Previous Work:
Previous research has predominantly focused on semantic similarity in English, leaving a gap in multilingual semantic relatedness evaluation. Prior methodologies may not have fully leveraged recent advancements in Transformer-based architectures for diverse multilingual datasets. Existing models might not effectively handle broader commonalities like topic, viewpoint, or temporal context across different languages.
Examples: N/A

You may only use Python and can only use the following external libraries in your code:
numpy, pandas, matplotlib, seaborn, torch, torchvision, tensorflow, keras, transformers, datasets, sklearn

Follow these instructions and do not forget them:
- First, come up with a high level plan based on your understanding of the problem and available tools and record it in the Research Plan and Status. You can revise the plan later.
- Research Plan and Status should well organized and succinctly keep track of 1) high level plan (can be revised), 2) what steps have been done and what steps are in progress, 3) short results and conclusions of each step after it has been performed. 
- Research Plan and Status must only include progress that has been made by previous steps. It should not include results not directly confirmed by the previous observation. 
- Performance numbers and estimates can only be confirmed and included in the status by running the code and observing the output.
- You should come up with a good experiment design that addresses the problem, and whenever applicable, define and measure the baseline performance of the relevant system or model before attempting any improvements.
- Follow the plan and try to achieve the goal as straightforwardly as possible.
- Highlight the supporting experiment results and reasoning before drawing any conclusions. 
- Do not try installing any new packages or libraries.
- If you believe you have solved the problem, you can use the Final Answer action to submit your answer. You can only submit once, so double check that you have achieved the goal before submitting.

Always respond in this format exactly:
Reflection: What does the observation mean? If there is an error, what caused the error and how to debug?
Research Plan and Status: The full high level research plan, with current status and confirmed results of each step briefly annotated. It must only include progress that has been made by previous steps. If there is any update, enclose the new update text in double asterisks **like this**. If there is no update, just copy the previous step Research Plan and Status. The high level plan from the previous step should be fully retained, unless it is intentionally revised.
Fact Check: List all objective statements in the updates to Research Plan and Status one by one and point out whether it is guessed versus directly confirmed by the previous observation directly above. Performance numbers can only be confirmed by running the code and observing the output.
Thought: What you are currently doing, what actions to perform and why
Action: the action to take, should be one of the names of the tools
Action Input: the input to the action as a valid JSON string
Observation: 
the result of the action
Given a training script on a dataset train.py, improve upon the current model performance (trained with current hyperparmeters in train.py).
data is in env/data/eng
"""
response = openai.Completion.create(
        engine="davinci-codex",
        prompt=user_message,
        max_tokens=300,
        temperature=0.5,
    )

script_content = response.choices[0].text.strip()
print(script_content)
# assistant_response = call_gpt4_api(user_message)
