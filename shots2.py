import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from evaluate import load
from googletrans import Translator
import dill

# Set caching to use dill
import datasets
datasets.config.TORCH_SAVE_SERIALIZATION = "dill"

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset (small subset for speed)
dataset = load_dataset("facebook/xnli", "ar")
dataset = dataset["train"].select(range(100))  # Use only 100 samples

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=3).to(device)

# Initialize translator globally
translator = Translator()

# Function to self-translate text
def self_translate(text, src_lang="ar", target_lang="en"):
    translated_text = translator.translate(text, src=src_lang, dest=target_lang).text
    back_translated_text = translator.translate(translated_text, src=target_lang, dest=src_lang).text
    return back_translated_text

# Preprocessing function (ensures serialization)
def preprocess_function(examples):
    examples["premise"] = [self_translate(p) for p in examples["premise"]]
    examples["hypothesis"] = [self_translate(h) for h in examples["hypothesis"]]
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length")

# Tokenize dataset
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,  # Mixed precision training
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    eval_dataset=encoded_dataset,
)

# Compute accuracy
metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)
    return metric.compute(predictions=predictions, references=labels)

trainer.compute_metrics = compute_metrics

# Evaluate Zero-Shot
print("\n=== Zero-Shot Evaluation ===")
eval_results = trainer.evaluate()
print(f"Zero-Shot Accuracy: {eval_results['eval_accuracy']:.4f}")

# Fine-Tuning and Comparing Shots
shot_accuracies = {}
for shot in [1, 2, 100]:
    print(f"\n=== {shot}-Shot Fine-Tuning ===")
    
    # Reload model before training
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=3).to(device)
    trainer.model = model

    # Select dataset with limited shots
    trainer.train_dataset = encoded_dataset.select(range(shot))
    
    # Fine-tune
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    shot_accuracies[shot] = eval_results['eval_accuracy']
    print(f"{shot}-Shot Accuracy: {eval_results['eval_accuracy']:.4f}")

# Final Accuracy Comparison
print("\n=== Final Accuracy Comparison ===")
for shot, acc in shot_accuracies.items():
    print(f"{shot}-Shot Accuracy: {acc:.4f}")
