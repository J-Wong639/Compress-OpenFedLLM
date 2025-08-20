from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm


from sklearn.datasets import fetch_20newsgroups
from datasets import Dataset

# 设置参数

model_path = "/mnt/sdb/huangjunlin/Compress-OpenFedLLM/output/COLA_8550_fedadam_c5s5_i30_b32a1_l512_r16a32_qlevel30_topk_ratio0.1_fpb_topk_sft_clientPQ_1e-3_1e-4_COLA_roberta_batchsize32_sub32_savemodel_20250702222517/checkpoint-200"  # 替换为你的模型路径
batch_size = 1  # 可自定义batch size


dataset = load_dataset("iambestfeed/glue-cola", split="validation")


dataset = dataset.remove_columns(["idx"])
dataset = dataset.rename_column("sentence", "instruction")
dataset = dataset.rename_column("label", "response")

# raw_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
# dataset = Dataset.from_dict({
#     "text": raw_data.data,
#     "label": raw_data.target})
# label_names = raw_data.target_names
# def format_20ng(example):
#     example["instruction"] = "Please categorize the following news\n\n" + example["text"]
#     example["response"] = label_names[example["label"]]
#     return example
# dataset = dataset.map(format_20ng, desc="Formatting 20NG train dataset to instruction-response")
# formatted_data = dataset.remove_columns(["text", "label"])

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()  # 设置为评估模式

# 使用tokenizer处理数据集
def preprocess_function(examples):
    return tokenizer(examples["instruction"], truncation=True, padding=True, max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 转换为 PyTorch dataset
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "response"])

# 创建 DataLoader
test_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size)

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 开始评估
correct = 0
total = 0

for batch in tqdm(test_dataloader):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["response"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Evaluation Accuracy: {accuracy:.4f}")



