
from datasets import load_dataset
import torch
import pandas as pd
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer, TrainingArguments
from accelerate import Accelerator

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

if __name__ == "__main__":
    imdb = load_dataset("imdb")

    # Preprocess data
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenized_imdb = imdb.map(tokenize_function, batched=True)
    tokenized_imdb.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Define model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Create an Accelerator instance
    accelerator = Accelerator()

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb['train'],
        eval_dataset=tokenized_imdb['test'],
    )

    # Prepare the trainer and data for accelerated training
    trainer, train_dataloader, eval_dataloader = accelerator.prepare(
        trainer, 
        DataLoader(tokenized_imdb['train'], batch_size=training_args.per_device_train_batch_size),
        DataLoader(tokenized_imdb['test'], batch_size=training_args.per_device_eval_batch_size)
    )

    # Start training
    trainer.train()

    # Use the Trainer's predict method to get predictions
    predictions = trainer.predict(tokenized_imdb['test'])

    # Extract the logits from the predictions
    logits = predictions.predictions

    # Convert logits to probabilities using softmax
    probs = torch.nn.functional.softmax(torch.Tensor(logits), dim=1).numpy()

    # Create a DataFrame with the probabilities
    submission = pd.DataFrame(probs, columns=['negative', 'positive'])

    # Save the DataFrame to a CSV file
    submission.to_csv('submission.csv', index=False)

    # Print a message confirming the save
    print('Predictions saved to submission.csv')

    # The following loop has been commented out as requested
    '''
    #evaluate model and print accuracy on test set, also save the predictions of probabilities per class to submission.csv
    submission = pd.DataFrame(columns=list(range(2)), index=range(len(imdb["test"])))
    acc = 0
    for idx, data in enumerate(imdb["test"]):
        text = data["text"]
        label = data["label"]
        pred = model(text) # TODO: replace with proper prediction
        pred = torch.softmax(pred, dim=0)
        submission.loc[idx] = pred.tolist()
        acc += int(torch.argmax(pred).item() == label)
    print("Accuracy: ", acc/len(imdb["test"]))
    
    submission.to_csv('submission.csv', index_label='idx')
    '''
