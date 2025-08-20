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

# # Step 1: 加载验证数据
# def load_jsonl(path):
#     with open(path, "r") as f:
#         lines = [json.loads(line) for line in f]
#     return lines
raw_eval_data = load_dataset(dataset_name, split="validation") # 看testing accuracy
# raw_eval_data = load_dataset(dataset_name, split="train") # 看training accuracy
 

# def format_rte(example):
#     example["instruction"] = f"Sentence 1: {example['text1']} \n Sentence 2: {example['text2']}"
#     example["response"] = "entailment" if example["label"] == 0 else "not entailment"
#     return example
# dataset = dataset.map(format_rte, desc=f"Formatting {dataset_name} to instruction-response.")
# dataset = dataset.remove_columns(["text1", "text2", "label", "idx", "label_text"])



# Step 2: 格式化为instruction-response风格
def format_rte(example):
    return {
        "instruction": f"Sentence 1: {example['text1']} \n Sentence 2: {example['text2']}",
        "response": "entailment" if example["label"] == 0 else "not entailment",
    }

formatted_data = list(map(format_rte, raw_eval_data))


# print(formatted_data[3])
# input()

# Step 3: Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

def tokenize(example):
    return tokenizer(example["instruction"], truncation=True, padding=False)

tokenized_dataset = list(map(tokenize, formatted_data))
labels = [ex["response"] for ex in formatted_data]
for i, example in enumerate(tokenized_dataset):
    example["response"] = labels[i]

# Step 4: 转换为 PyTorch DataLoader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
eval_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)

# Step 5: 加载模型
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
model.cuda()

# Step 6: Evaluation Loop
correct = 0
total = 0
for batch in tqdm(eval_dataloader):
    batch = {k: v.cuda() for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    predictions = torch.argmax(outputs.logits, dim=-1)
    correct += (predictions == batch["labels"]).sum().item()
    total += batch["labels"].size(0)

accuracy = correct / total
print(f"Evaluation Accuracy: {accuracy:.4f}")
