import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("xnli", "ar")  # Change "en" to your target language

# Limit training data to 4000 samples
dataset["train"] = dataset["train"].select(range(4000))

# Load tokenizer and model
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenization function
def preprocess_data(example):
    encoding = tokenizer(
        example["premise"], example["hypothesis"],
        padding="max_length", truncation=True, max_length=64  # Reduced for speed
    )
    encoding["labels"] = example["label"]
    return encoding

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Convert to PyTorch dataset
train_dataset = TensorDataset(
    torch.tensor(tokenized_dataset["train"]["input_ids"]),
    torch.tensor(tokenized_dataset["train"]["attention_mask"]),
    torch.tensor(tokenized_dataset["train"]["labels"])
)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):  # Adjust epochs as needed
    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed!")

print("Training finished!")



from sklearn.metrics import accuracy_score

# Tokenize the test dataset
tokenized_test = dataset["validation"].map(preprocess_data, batched=True)

# Convert to PyTorch dataset
test_dataset = TensorDataset(
    torch.tensor(tokenized_test["input_ids"]),
    torch.tensor(tokenized_test["attention_mask"]),
    torch.tensor(tokenized_test["labels"])
)

# Create DataLoader
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Move model to evaluation mode
model.eval()

# Evaluate model
all_preds = []
all_labels = []

with torch.no_grad():  # Disable gradient calculation
    for batch in test_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)  # Get highest probability class

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")
