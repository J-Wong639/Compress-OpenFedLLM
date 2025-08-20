from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from peft import get_peft_model, LoraConfig, TaskType
import torch


# 启用更快的 matmul 精度配置
torch.set_float32_matmul_precision('high')

 
# Dataset id from huggingface.co/dataset
dataset_id = "SetFit/rte"
 
# Load raw dataset
train_dataset = load_dataset(dataset_id, split='train')
test_dataset = load_dataset(dataset_id, split='validation')


def format_rte(example):
    example["text"] = f"Does Sentence 1 entail Sentence 2?\nSentence 1: {example['text1']} \n Sentence 2: {example['text2']}"
    example["label"] = 0 if example["label"] == 0 else 1
    return example
train_dataset = train_dataset.map(format_rte, desc=f"Formatting {dataset_id} to text-label.")
test_dataset = test_dataset.map(format_rte, desc=f"Formatting {dataset_id} to text-label.")
train_dataset = train_dataset.remove_columns(["text1", "text2", "idx", "label_text"])
test_dataset = test_dataset.remove_columns(["text1", "text2", "idx", "label_text"])


# import pdb; pdb.set_trace()


from transformers import AutoTokenizer
 

model_id = "answerdotai/ModernBERT-base"


tokenizer = AutoTokenizer.from_pretrained(model_id)
 

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt", padding_side='left')
 

if "label" in train_dataset.features.keys():
    train_dataset =  train_dataset.rename_column("label", "labels") # to match Trainer
tokenized_train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])

if "label" in test_dataset.features.keys():
    test_dataset =  test_dataset.rename_column("label", "labels") # to match Trainer
tokenized_test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=["text"])







from transformers import AutoModelForSequenceClassification

model_id = "answerdotai/ModernBERT-base"
 


from datasets import ClassLabel

label_names = ["entailment", "not entailment"]
class_label = ClassLabel(names=label_names)


tokenized_train_dataset = tokenized_train_dataset.cast_column("labels", class_label)
tokenized_test_dataset = tokenized_test_dataset.cast_column("labels", class_label)

labels = tokenized_train_dataset.features["labels"].names
num_labels = len(labels)

label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label


model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=num_labels, label2id=label2id, id2label=id2label,
)

# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Linear):
#         print(name)

# import pdb; pdb.set_trace()

peft_config = LoraConfig(
    task_type = TaskType.SEQ_CLS,
    inference_mode = False,
    r = 16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]
)
model= get_peft_model(model, peft_config)





import numpy as np
from sklearn.metrics import f1_score
 

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(
            labels, predictions, labels=labels, pos_label=1, average="weighted"
        )
    return {"f1": float(score) if score == 1 else score}



from huggingface_hub import HfFolder
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir= "modern-bert3",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-4,
    num_train_epochs=50,
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
    max_grad_norm=1.0,
)
 


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

train_results = trainer.train()

eval_results = trainer.evaluate(eval_dataset=tokenized_train_dataset)

print(train_results)
print(eval_results)


eval_results= trainer.evaluate()
print(eval_results)



from peft import PeftModel

trained_model=trainer.model

trained_model = trained_model.merge_and_unload()

trained_model.save_pretrained("modern-bert3-lora-merged_50epoch")
tokenizer.save_pretrained("modern-bert3-lora-merged_50epoch")













