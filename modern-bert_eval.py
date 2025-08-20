from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("modern-bert3-lora-merged")
tokenizer = AutoTokenizer.from_pretrained("modern-bert3-lora-merged")


from datasets import load_dataset

# 重新加载 RTE 测试集
raw_test_dataset = load_dataset("SetFit/rte", split="validation")

# 格式化
def format_rte(example):
    example["text"] = f"Does Sentence 1 entail Sentence 2?\nSentence 1: {example['text1']} \n Sentence 2: {example['text2']}"
    example["label"] = 0 if example["label"] == 0 else 1
    return example

formatted_test = raw_test_dataset.map(format_rte)
formatted_test = formatted_test.rename_column("label", "labels")
formatted_test = formatted_test.remove_columns(["text1", "text2", "idx", "label_text"])

# Tokenize
tokenized_test_dataset = formatted_test.map(
    lambda batch: tokenizer(batch["text"], padding=True, truncation=True),
    batched=True,
    remove_columns=["text"]
)


from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
from sklearn.metrics import f1_score

# 定义 metrics 函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"f1": f1_score(labels, predictions, average="weighted")}

# 构造 dummy training args（不影响评估）
training_args = TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=16)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# 执行评估
eval_results = trainer.evaluate()
print(eval_results)
