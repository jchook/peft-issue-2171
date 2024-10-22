import torch
from torch.utils.data import DataLoader
from peft import PeftModelForSequenceClassification, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import evaluate
from tqdm import tqdm

peft_model_id = "./output/lora_epoch_1"
dataset_name = "imdb"
batch_size = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
config = PeftConfig.from_pretrained(peft_model_id)
base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
inference_model = PeftModelForSequenceClassification.from_pretrained(base_model, peft_model_id)

def tokenize_fn(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# Load datasets
print("Loading dataset...")
dataset = load_dataset(dataset_name)

# Select only a few examples for faster testing
dataset['train'] = dataset['train'].select(range(1))
dataset['test'] = dataset['test'].select(range(100))
dataset['unsupervised'] = dataset['unsupervised'].select(range(1))

# Tokenize datasets
print("Tokenizing datasets...")
tokenized_datasets = dataset.map(tokenize_fn, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')

# Prepare dataloader
eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=batch_size, shuffle=False)

# Reset the metric
metric = evaluate.load("accuracy")

# Prepare the model for evaluation
inference_model.to(device)
inference_model.eval()

# Test loop
for step, batch in enumerate(tqdm(eval_dataloader)):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = inference_model(**batch)
    predictions = outputs.logits.argmax(dim=-1)
    predictions, references = predictions, batch["labels"]
    metric.add_batch(
        predictions=predictions,
        references=references,
    )

eval_metric = metric.compute()
print(eval_metric)
