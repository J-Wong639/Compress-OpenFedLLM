from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

# 设置参数
dataset_name = "SetFit/rte"
model_path = "/mnt/sdb/huangjunlin/Compress-OpenFedLLM/output/rte_2490_fedavg_c1s1_i5_b32a1_l512_r16a32_qlevel30_topk_ratio0.1_fpb_center_fedavg_20250801154443/full-100"  # 替换为你的模型路径
batch_size = 16  # 可自定义batch size

# Load validation data
# raw_eval_data = load_dataset(dataset_name, split="validation") # 看testing accuracy
raw_eval_data = load_dataset(dataset_name, split="train") # 看training accuracy

# Step 2: 格式化为instruction-response风格
def format_rte(example):
    return {
        "instruction": f"Does the first sentence entail the second one?\nSentence 1: {example['text1']}\nSentence 2: {example['text2']}",
        "label": 1 if example["label"] == 1 else 0,
    }

formatted_data = list(map(format_rte, raw_eval_data))

# Step 3: Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

def tokenize(example):
    return tokenizer(example["instruction"], truncation=True, padding=False)

tokenized_dataset = list(map(tokenize, formatted_data))
labels = [ex["label"] for ex in formatted_data]
for i, example in enumerate(tokenized_dataset):
    example["label"] = labels[i]

# Step 4: 转换为 PyTorch DataLoader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
eval_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)

# Step 5: 加载模型
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
model.cuda()

# Step 6: Evaluation Loop with loss calculation
correct = 0
total = 0
total_loss = 0.0

# We'll use CrossEntropyLoss which is what's typically used for classification
loss_fn = torch.nn.CrossEntropyLoss()

for batch in tqdm(eval_dataloader):
    batch = {k: v.cuda() for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    
    # Calculate loss
    loss = loss_fn(outputs.logits, batch["labels"])
    total_loss += loss.item()
    
    # Calculate accuracy
    predictions = torch.argmax(outputs.logits, dim=-1)
    correct += (predictions == batch["labels"]).sum().item()
    total += batch["labels"].size(0)

accuracy = correct / total
average_loss = total_loss / len(eval_dataloader)
print(f"Training Accuracy: {accuracy:.4f}")
print(f"Training Loss: {average_loss:.4f}")