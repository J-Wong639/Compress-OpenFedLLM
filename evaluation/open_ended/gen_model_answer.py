import datasets
import argparse
import json
import sys
sys.path.append("../../")
from tqdm import tqdm
import os
import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
# from vllm import LLM, SamplingParams

from utils.template import TEMPLATE_DICT

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--template", type=str, default="alpaca")
parser.add_argument("--use_vllm", action="store_true", default=False)
parser.add_argument("--bench_name", type=str, default="vicuna")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
args = parser.parse_args()
print(args)

# 检查是否同时使用了VLLM和LORA,这是不允许的组合
# 因为VLLM不支持直接使用LORA权重,需要先合并LORA权重到基础模型中
if args.use_vllm and args.lora_path is not None:
    raise ValueError("Cannot use both VLLM and LORA, need to merge the lora and then use VLLM")

# 从模板字典中获取指定模板的第一个元素
# args.template指定使用哪个模板(如"alpaca"),TEMPLATE_DICT存储了所有可用的模板
template = TEMPLATE_DICT[args.template][0]

# 打印当前使用的模板信息
print(f">> You are using template: {template}")

# ============= Load dataset =============
# 根据不同的benchmark名称加载不同的数据集
if args.bench_name == "alpaca":
    # 加载alpaca评估数据集,获取其中的eval部分
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    # 因为alpaca_eval数据集里本来就有instruction特征，所以不用转换
    # 设置生成token的最大长度为2048
    max_new_tokens = 2048
elif args.bench_name == "vicuna":
    # 从本地json文件加载vicuna数据集的训练部分
    eval_set = datasets.load_dataset("json", data_files="data/vicuna/question.jsonl")['train']
    # 定义重命名函数,将turns[0]的内容赋值给instruction字段
    def rename(example):
        # turns[0]指的是对话轮次中的第一个问题
        # 例如: {'turns': ['如何提高时间管理能力?'], 'question_id': 1} 中的 '如何提高时间管理能力?'
        example['instruction'] = example['turns'][0]
        return example
    # 对数据集应用重命名函数
    # 对数据集中的每一条数据应用rename函数
    # rename函数会将turns[0]的内容赋值给instruction字段
    # .map()会遍历数据集中的每一个样本,对每个样本执行rename函数进行转换
    # 举例:
    # 原始数据: {'turns': ['如何提高时间管理能力?'], 'question_id': 1, 'category': 'generic'}
    # 转换后: {'turns': ['如何提高时间管理能力?'], 'question_id': 1, 'category': 'generic', 'instruction': '如何提高时间管理能力?'}
    # 对eval_set数据集中的每一条数据应用rename函数
    # rename函数会将turns[0]的内容赋值给instruction字段
    # 例如:
    # 输入: {'turns': ['如何提高时间管理能力?'], 'question_id': 1}
    # 输出: {'turns': ['如何提高时间管理能力?'], 'question_id': 1, 'instruction': '如何提高时间管理能力?'}
    # 对eval_set数据集中的每一条数据应用rename函数
    # 转换前的数据格式示例:
    # {'turns': ['如何提高时间管理能力?'], 'question_id': 1, 'category': 'generic'}
    # 转换后的数据格式示例:
    # {'turns': ['如何提高时间管理能力?'], 'question_id': 1, 'category': 'generic', 'instruction': '如何提高时间管理能力?'}
    # .map()函数是数据集处理中的一个重要方法
    # 它会遍历数据集中的每一个样本,对每个样本执行指定的函数进行转换
    # 在这里,map(rename)会对eval_set中的每条数据应用rename函数
    # rename函数会将turns[0]的内容赋值给instruction字段
    eval_set = eval_set.map(rename)
    # 设置生成token的最大长度为2048
    max_new_tokens = 2048
elif args.bench_name == "advbench":
    # 从本地csv文件加载advbench数据集的训练部分
    eval_set = datasets.load_dataset("csv", data_files="data/advbench/advbench.csv")["train"]
    # 将goal列重命名为instruction
    eval_set = eval_set.rename_column("goal", "instruction")
    # 移除target列
    eval_set = eval_set.remove_columns(["target"])
    # 设置生成token的最大长度为1024
    max_new_tokens = 1024
else:
    # 如果benchmark名称无效则抛出异常
    raise ValueError("Invalid benchmark name")

# ============= Extract model name from the path. The name is used for saving results. =============
if args.lora_path:
    pre_str, checkpoint_str = os.path.split(args.lora_path)
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = checkpoint_str.split("-")[-1]
    model_name = f"{exp_name}_{checkpoint_id}"
else:
    pre_str, last_str = os.path.split(args.base_model_path)
    if last_str.startswith("full"):                 # if the model is merged as full model
        _, exp_name = os.path.split(pre_str)
        checkpoint_id = last_str.split("-")[-1]
        model_name = f"{exp_name}_{checkpoint_id}"
    else:
        model_name = last_str                       # mainly for base model

# ============= Load previous results if exists =============
result_path = f"./data/{args.bench_name}/model_answer/{model_name}.json"

if os.path.exists(result_path):
    with open(result_path, "r") as f:
        result_list = json.load(f)
else:
    result_list = []
existing_len = len(result_list)
print(f">> Existing length: {existing_len}")

# ============= Generate responses =============
if args.use_vllm:
    model = LLM(model=args.base_model_path)
    if args.bench_name == "advbench":
        input_list = [template.format(example["instruction"]+'.', "", "")[:-1] for example in eval_set]
    else:
        input_list = [template.format(example["instruction"], "", "")[:-1] for example in eval_set] # TODO: use fastchat conversation
    input_list = input_list[existing_len:]
    print(f">> Example input: {input_list[0]}")
    sampling_params = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=max_new_tokens)
    generations = model.generate(input_list, sampling_params)
    generations = [generation.outputs[0].text for generation in generations]

    for i, example in tqdm(enumerate(eval_set)):
        if i < existing_len:
            continue
        example['output'] = generations[i-existing_len]
        example['generator'] = exp_name
        result_list.append(example)
    with open(result_path, "w") as f:
        json.dump(result_list, f, indent=4)

else:
    # 设置设备为CUDA以使用GPU
    device = 'cuda'
    # 加载预训练的因果语言模型,使用float16精度以节省显存
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16).to(device)
    # 如果指定了LoRA路径,则加载LoRA权重
    if args.lora_path is not None:
        model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype=torch.float16).to(device)
    # 加载对应的分词器
    # 分词器(tokenizer)用于将文本转换为模型可以理解的数字序列
    # 例如:"我爱中国" -> [101, 2769, 4263, 1744, 1249, 102]
    # 或者:"Hello World" -> [101, 7592, 2088, 102]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    # 遍历评估数据集
    # tqdm是一个进度条库,用于显示循环的进度
    # 它会在循环执行时显示一个进度条,包含完成百分比、已用时间、预计剩余时间等信息
    # enumerate(eval_set)会遍历eval_set并返回索引i和元素example
    # tqdm包装这个遍历过程,添加进度显示功能
    for i, example in tqdm(enumerate(eval_set)):
        # 跳过已经处理过的样本
        if i < existing_len:
            continue
        # 根据不同的benchmark类型构造输入指令
        if args.bench_name == "advbench":
            instruction = template.format(example["instruction"]+'.', "", "")[:-1]
        else:
            instruction = template.format(example["instruction"], "", "")[:-1]      # TODO: use fastchat conversation
        # 将输入文本转换为模型可处理的token ids
        input_ids = tokenizer.encode(instruction, return_tensors="pt").to(device)
        # 生成回复,设置采样参数
        output_ids = model.generate(inputs=input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=1.0, temperature=0.7)
        # 只保留新生成的token ids
        output_ids = output_ids[0][len(input_ids[0]):]
        # 将token ids解码为文本
        result = tokenizer.decode(output_ids, skip_special_tokens=True)
        # 保存生成结果
        example['output'] = result
        example['generator'] = model_name

        # 打印输入输出信息
        print(f"\nInput: \n{instruction}".encode('utf-8'))
        print(f"\nOutput: \n{result}".encode('utf-8'))
        print("="*100)
        # 将结果添加到结果列表
        result_list.append(example)
        # 将更新后的结果保存到文件
        with open(result_path, "w") as f:
            json.dump(result_list, f, indent=4)

print(f">> You are using template: {template}")
