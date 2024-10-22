import torch
from torch.optim import AdamW
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import evaluate
import numpy as np

# Training config
base_model_name = "microsoft/deberta-v3-base"
dataset_name = "imdb"
output_dir = "output"
batch_size = 4
num_epochs = 2 # important that this is > 1
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

# PEFT config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    lora_dropout=0.1,
)

# Base model and tokenizer
print("Loading base model...")
model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print("Applying LoRA to base model...")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

def tokenize_fn(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# Tokenize datasets
print("Loading dataset...")
dataset = load_dataset(dataset_name)

# Select only a few examples for faster training / testing
dataset['train'] = dataset['train'].select(range(500))
dataset['test'] = dataset['test'].select(range(200))
dataset['unsupervised'] = dataset['unsupervised'].select(range(1))

# Tokenize datasets
print("Tokenizing datasets...")
tokenized_datasets = dataset.map(tokenize_fn, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')

# Prepare dataloaders
train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=batch_size, shuffle=False)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = (num_epochs * len(train_dataloader))

# Learning rate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

# Metrics
metric = evaluate.load("accuracy")

# Move model to GPU
print(f"Moving model to {device}...")
model.to(device)

# Training loop
print("Starting training loop...")
smoothed_accuracy = []
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        predictions = torch.argmax(outputs.logits, dim=-1)
        accuracy = (predictions == batch["labels"]).float().mean().item()
        smoothed_accuracy.append(accuracy)

        if len(smoothed_accuracy) > 100:
            smoothed_accuracy.pop(0)

        if (step + 1) % 10 == 0:
            smooth_acc = np.mean(smoothed_accuracy)
            print(f" acc {smooth_acc}")

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(predictions=predictions, references=references)

    # Compute and print metrics
    eval_metric = metric.compute()
    print(f"epoch {epoch+1}:", eval_metric)

    # Save the model after each epoch
    model.save_pretrained(f"{output_dir}/lora_epoch_{epoch+1}")

