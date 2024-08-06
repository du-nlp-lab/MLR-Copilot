
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel, AdamW

DIMENSIONS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
SEED = 42
BATCH_SIZE = 16
MAX_LEN = 512
EPOCHS = 3

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

def compute_metrics_for_regression(y_test, y_test_pred):
    metrics = {}
    for task in DIMENSIONS:
        targets_task = [t[DIMENSIONS.index(task)] for t in y_test]
        pred_task = [l[DIMENSIONS.index(task)] for l in y_test_pred]
        
        rmse = mean_squared_error(targets_task, pred_task, squared=False)

        metrics[f"rmse_{task}"] = rmse
    
    return metrics

class DistilBertRegressor(nn.Module):
    def __init__(self, distilbert_model):
        super(DistilBertRegressor, self).__init__()
        self.distilbert = distilbert_model
        self.regressor = nn.Linear(768, len(DIMENSIONS))

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]
        return self.regressor(hidden_state)

def train_model(X_train, y_train, X_valid, y_valid):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model = DistilBertRegressor(distilbert_model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LEN)
    valid_encodings = tokenizer(X_valid, truncation=True, padding=True, max_length=MAX_LEN)

    train_dataset = TensorDataset(
        torch.tensor(train_encodings['input_ids']),
        torch.tensor(train_encodings['attention_mask']),
        torch.tensor(y_train, dtype=torch.float)
    )
    valid_dataset = TensorDataset(
        torch.tensor(valid_encodings['input_ids']),
        torch.tensor(valid_encodings['attention_mask']),
        torch.tensor(y_valid, dtype=torch.float)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask)
                val_loss += loss_fn(outputs, labels).item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
    
    model.load_state_dict(best_model)
    return model, tokenizer

def predict(model, tokenizer, texts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors='pt')
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    with torch.no_grad():
outputs = model(input_ids, attention_mask)
    
    return outputs.cpu().numpy()

# Main section
if __name__ == "__main__":
    # Load your data here
    # X_train, X_valid, y_train, y_valid = ...

    # Train the model
    model, tokenizer = train_model(X_train, y_train, X_valid, y_valid)

    # Make predictions
    predictions = predict(model, tokenizer, X_valid)

    # Compute metrics
    metrics = compute_metrics_for_regression(y_valid, predictions)
    print(metrics)

    return model, tokenizer

def predict(model, tokenizer, X):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    encodings = tokenizer(X, truncation=True, padding=True, max_length=MAX_LEN)
    dataset = TensorDataset(
        torch.tensor(encodings['input_ids']),
        torch.tensor(encodings['attention_mask'])
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask)
            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions)

if __name__ == '__main__':
    ellipse_df = pd.read_csv('train.csv', 
                            header=0, names=['text_id', 'full_text', 'Cohesion', 'Syntax', 
                            'Vocabulary', 'Phraseology','Grammar', 'Conventions'], 
                            index_col='text_id')
    ellipse_df = ellipse_df.dropna(axis=0)

    data_df = ellipse_df
    X = list(data_df.full_text.to_numpy())
    y = np.array([data_df.drop(['full_text'], axis=1).iloc[i] for i in range(len(X))])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    model, tokenizer = train_model(X_train, y_train, X_valid, y_valid)

    y_valid_pred = predict(model, tokenizer, X_valid)
    metrics = compute_metrics_for_regression(y_valid, y_valid_pred)
    print(metrics)
    print("final MCRMSE on validation set: ", np.mean(list(metrics.values())))

    submission_df = pd.read_csv('test.csv',  header=0, names=['text_id', 'full_text'], index_col='text_id')
    X_submission = list(submission_df.full_text.to_numpy())
    y_submission = predict(model, tokenizer, X_submission)
    submission_df = pd.DataFrame(y_submission, columns=DIMENSIONS)
    submission_df.index = submission_df.index.rename('text_id')
    submission_df.to_csv('submission.csv')
