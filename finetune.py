"""
Script for fine-tuning a T5 model on a disaster classification dataset using Hugging Face Transformers.
"""

# Install necessary packages
!pip install -U transformers datasets huggingface_hub tensorboard==2.11 accelerate

# Install Git Large File Storage
!sudo apt-get install git-lfs --yes

import re
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    T5ForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig
)
from huggingface_hub import HfFolder, notebook_login
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

# Login to Hugging Face
notebook_login()

# Define model and dataset identifiers
model_id = "google-t5/t5-small"
dataset_name = "kaikouraEarthquake"
dataset_id = f"Piyyussh/{dataset_name}"
repository_id = f"rizvi-rahil786/t5-small-{dataset_name}"  # Your Hugging Face model repo

# Define noise words to clean from the dataset
Noise = [
    'RT', '-', '/', '22w', 'w/', 'f', 'd', '--', 'l', 'w-n-w', 'e', 'ir', 'm',
    'x', 't', 'b', 'r', '1/', 'h', 'wx', 'cc', 'fr', 'w', 'http/', 's', 'http//',
    'htt', '"', 'ct', 'g', 'n', '...', 'ndrrmc', 'http:/'
]

def Preprocessing(text):
    """
    Clean the input text by removing URLs, Twitter handles, special characters, and lowercasing.
    """
    text = re.sub('http://\S+|https://\S+', '', text)
    text = " ".join(filter(lambda x: x[0] != '@', text.split()))
    text = re.sub("[\(\[].*?[\)\]]", "", text)
    word_tokens = text.split()
    text = ' '.join([w for w in word_tokens if w not in Noise])
    return text.lower()

# Load and encode dataset
dataset = load_dataset(dataset_id)
dataset = dataset.class_encode_column("label")
train_dataset = dataset['train']
test_dataset = dataset['test']
val_dataset = dataset['train']  # Optional: use same data for validation if split not provided

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize(batch):
    """
    Tokenize a batch of text data with padding and truncation.
    """
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=256)

# Apply tokenization
train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

# Format datasets for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Extract label information
num_labels = dataset['train'].features['label'].num_classes
class_names = dataset["train"].features["label"].names
print(f"Number of labels: {num_labels}")
print(f"The labels: {class_names}")

# Create id2label mapping
id2label = {i: label for i, label in enumerate(class_names)}

# Update model configuration with label mapping
config = AutoConfig.from_pretrained(model_id)
config.update({"id2label": id2label})

# Load model with updated config
model = T5ForSequenceClassification.from_pretrained(model_id, config=config)

# Define training arguments
training_args = TrainingArguments(
    output_dir=repository_id,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="tensorboard",
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=repository_id,
    hub_token=HfFolder.get_token(),
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start training
trainer.train()

# Save tokenizer and model card, then push to hub
tokenizer.save_pretrained(repository_id)
trainer.create_model_card()
trainer.push_to_hub()

# Load pipeline for inference
pipe = pipeline("text-classification", model=repository_id)

# Example inference
text = "Its one thing to put yourself at risk, but to risk the lives of children and emergency personnel is unconscionable."
result = pipe(text)
predicted_label = result[0]["label"]
print(f"Predicted label: {predicted_label}")

# Batch evaluation on test set
dataset = load_dataset(dataset_id)
y_pred = []

for i, row in enumerate(dataset["test"]):
    result = pipe(row['text'])
    print(f"Processing test example {i}")
    y_pred.append(result[0]['label'])

# Print predictions and examples
for row in dataset['test']:
    print(row)
print("Predictions:", y_pred)

# Evaluate performance
y_true = [class_names[label] for label in dataset['test']['label']]

accuracy = accuracy_score(y_true, y_pred)
print('Accuracy:', accuracy)

print('F1 Score (macro):', f1_score(y_true, y_pred, average='macro'))
print("Cohen's Kappa:", cohen_kappa_score(y_true, y_pred))
print('F1 Score (weighted):', f1_score(y_true, y_pred, average='weighted'))
print('F1 Score (micro):', f1_score(y_true, y_pred, average='micro'))
