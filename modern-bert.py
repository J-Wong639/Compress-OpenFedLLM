from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
 
# Dataset id from huggingface.co/dataset
# dataset_id = "argilla/synthetic-domain-text-classification"
 
# # Load raw dataset
# train_dataset = load_dataset(dataset_id, split='train')

# split_dataset = train_dataset.train_test_split(test_size=0.1)
# split_dataset['train'][0]
# {'text': 'Recently, there has been an increase in property values within the suburban areas of several cities due to improvements in infrastructure and lifestyle amenities such as parks, retail stores, and educational institutions nearby. Additionally, new housing developments are emerging, catering to different family needs with varying sizes and price ranges. These changes have influenced investment decisions for many looking to buy or sell properties.', 'label': 14}

# import pdb; pdb.set_trace()




from datasets import load_dataset


dataset_name="SetFit/rte"

# 加载RTE数据集
dataset = load_dataset(dataset_name, split="train")

def format_rte(example):
    example["label"] = "entailment" if example["label"] == 0 else "not entailment"
    example["text"] = f"Does Sentence 1 entail Sentence 2?\nSentence 1: {example['text1']} \n Sentence 2: {example['text2']}"
    return example
dataset = dataset.map(format_rte, desc=f"Formatting {dataset_name} to instruction-response.")
dataset = dataset.remove_columns(["text1", "text2", "idx", "label_text"])


split_dataset = dataset.train_test_split(test_size=0.1)

# print(type(dataset))




from transformers import AutoTokenizer
 
# Model id to load the tokenizer
model_id = "answerdotai/ModernBERT-base"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
 
# Tokenize helper function
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt")
 
# Tokenize dataset
if "label" in split_dataset["train"].features.keys():
    split_dataset =  split_dataset.rename_column("label", "labels") # to match Trainer
tokenized_dataset = split_dataset.map(tokenize, batched=True, remove_columns=["text"])
 
# print(tokenized_dataset["train"].features.keys())
# dict_keys(['labels', 'input_ids', 'attention_mask'])


from transformers import AutoModelForSequenceClassification
 
# Model id to load the tokenizer
model_id = "answerdotai/ModernBERT-base"
 
# Prepare model labels - useful for inference
labels = tokenized_dataset["train"].features["labels"]
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
 
# Download the model from huggingface.co/models
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=num_labels, label2id=label2id, id2label=id2label,
)


