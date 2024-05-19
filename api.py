import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

# Load your dataset
df = pd.read_csv('collected_data.csv')

# Split your data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['off-platform'], test_size=0.2, random_state=42)

# Convert labels to lists to ensure proper indexing
train_labels = train_labels.tolist()
test_labels = test_labels.tolist()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

# Create dataset class
class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ConversationDataset(train_encodings, train_labels)
test_dataset = ConversationDataset(test_encodings, test_labels)

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments with a temporary directory for logging
with tempfile.TemporaryDirectory() as tmp_dir:
    training_args = TrainingArguments(
        output_dir=tmp_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=None,  # Disable logging
        evaluation_strategy="no",  # Disable evaluation during training
        save_strategy="no",  # Disable model saving during training
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()
    print(results)

# Inference function
def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

# FastAPI setup
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

@app.post("/check-message")
async def check_message(message: Message):
    prediction = predict(message.text)
    result = "Off-platform" if prediction == 1 else "On-platform"
    return {"message": message.text, "classification": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
