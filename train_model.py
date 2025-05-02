import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

# Load your dataset (replace with your actual dataset path)
df = pd.read_csv("C:/Users/Swara/Desktop/Git_projects/flipkart_product.csv",encoding='ISO-8859-1')

# Preprocess the data (combining Review and Summary)
df["text"] = df["Review"].astype(str) + " " + df["Summary"].astype(str)
df.dropna(subset=["text", "Rate"], inplace=True)
df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
df["label"] = df["Rate"].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else None))
df.dropna(subset=["label"], inplace=True)
df["label"] = df["label"].astype(int)

# Split into train/test sets
train_df, test_df = train_test_split(df[["text", "label"]], test_size=0.2, random_state=42)

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns
train_dataset = train_dataset.remove_columns([col for col in ["text", "__index_level_0__"] if col in train_dataset.column_names])
test_dataset = test_dataset.remove_columns([col for col in ["text", "__index_level_0__"] if col in test_dataset.column_names])

# Set PyTorch format
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model/tokenizer")

print("Model training complete and saved!")
