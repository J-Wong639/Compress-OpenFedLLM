from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
 
# Dataset id from huggingface.co/dataset
dataset_id = "argilla/synthetic-domain-text-classification"
 
# Load raw dataset
train_dataset = load_dataset(dataset_id, split='train')

split_dataset = train_dataset.train_test_split(test_size=0.1)




from transformers import AutoTokenizer
 
# Model id to load the tokenizer
model_id = "answerdotai/ModernBERT-base"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
 
# Tokenize helper function
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt", padding_side="left")
 
# Tokenize dataset
if "label" in split_dataset["train"].features.keys():
    split_dataset =  split_dataset.rename_column("label", "labels") # to match Trainer
tokenized_dataset = split_dataset.map(tokenize, batched=True, remove_columns=["text"])
 
tokenized_dataset["train"].features.keys()
# dict_keys(['labels', 'input_ids', 'attention_mask'])



from transformers import AutoModelForSequenceClassification
 
# Model id to load the tokenizer
model_id = "answerdotai/ModernBERT-base"
 
# Prepare model labels - useful for inference
labels = tokenized_dataset["train"].features["labels"].names
num_labels = len(labels)

label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label



 
# Download the model from huggingface.co/models
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=num_labels, label2id=label2id, id2label=id2label,
)


import numpy as np
from sklearn.metrics import f1_score
 
# Metric helper method
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(
            labels, predictions, labels=labels, pos_label=1, average="weighted"
        )
    return {"f1": float(score) if score == 1 else score}



from huggingface_hub import HfFolder
from transformers import Trainer, TrainingArguments
 
# Define training args
training_args = TrainingArguments(
    output_dir= "Modern-bert2",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
        num_train_epochs=5,
    bf16=False, # bfloat16 training 
    optim="adamw_torch_fused", # improved optimizer 
    # logging & evaluation strategies
    logging_strategy="steps",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    # use_mps_device=True,
    metric_for_best_model="f1",
    # push to hub parameters
    push_to_hub=True,
    hub_strategy="every_save",
    hub_token=HfFolder.get_token(),
)
 
# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

results = trainer.train()

print(results.training_loss)
print(results)




