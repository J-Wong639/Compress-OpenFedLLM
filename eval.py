import copy
import os
from tqdm import tqdm
import numpy as np


from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
import torch
import transformers
from transformers import Trainer

import math
from datasets import load_dataset
import datasets

from FinNLP.finnlp.benchmarks.fpb import test_fpb
from FinNLP.finnlp.benchmarks.fiqa import test_fiqa , add_instructions
from FinNLP.finnlp.benchmarks.tfns import test_tfns
from FinNLP.finnlp.benchmarks.nwgi import test_nwgi
from utils.merge_lora import merge_lora
from transformers import LlamaTokenizerFast, LlamaForCausalLM
from peft import PeftModel


# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)


# ===== Get model config ===== 
device_map, quantization_config, torch_dtype = get_model_config(script_args)



# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)






# evaluation






# base_model="meta-llama/Llama-2-7b-hf"
# peft_model="output/fingpt-sentiment-train_10000_fedavg_c50s5_i10_b16a1_l512_r32a64_qlevel32_fpb_20241101134406/checkpoint-200"
# model = LlamaForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map = "cuda:0", load_in_8bit = True, torch_dtype=torch_dtype)
# model = PeftModel.from_pretrained(model, peft_model)
# model = torch.compile(model)  # Please comment this line if your platform does not support torch.compile
# model = model.eval()


model_path="output/fingpt-sentiment-train_10000_fedavgm_c50s5_i10_b16a1_l512_r32a64_qlevel32_fiqa_20241104140330/checkpoint-200"


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config, #stop pre-train quantization
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)
model.eval()


    
if fed_args.test_dataset is not None:
    if fed_args.test_dataset=='fpb':
        dataset, acc, f1_macro, f1_micro, f1_weighted = test_fpb(model, tokenizer, batch_size = fed_args.test_batch_size)
        Acc, F1_macro, F1_micro, F1_weighted=[], [], [], []
        Acc.append(acc)
        F1_macro.append(f1_macro)
        F1_micro.append(f1_micro)
        F1_weighted.append(f1_weighted)
        np.save(os.path.join(script_args.output_dir, "testing_acc.npy"), np.array(Acc))
        np.save(os.path.join(script_args.output_dir, "f1_macro.npy"), np.array(F1_macro))
        np.save(os.path.join(script_args.output_dir, "f1_micro.npy"), np.array(F1_micro))
        np.save(os.path.join(script_args.output_dir, "f1_weighted.npy"), np.array(F1_weighted))
    elif fed_args.test_dataset=='fiqa':
        dataset, acc, f1_macro, f1_micro, f1_weighted = test_fiqa(model, tokenizer, prompt_fun = add_instructions, batch_size = fed_args.test_batch_size)
        Acc, F1_macro, F1_micro, F1_weighted=[], [], [], []
        Acc.append(acc)
        F1_macro.append(f1_macro)
        F1_micro.append(f1_micro)
        F1_weighted.append(f1_weighted)
        np.save(os.path.join(script_args.output_dir, "testing_acc.npy"), np.array(Acc))
        np.save(os.path.join(script_args.output_dir, "f1_macro.npy"), np.array(F1_macro))
        np.save(os.path.join(script_args.output_dir, "f1_micro.npy"), np.array(F1_micro))
        np.save(os.path.join(script_args.output_dir, "f1_weighted.npy"), np.array(F1_weighted))
    elif fed_args.test_dataset=='tfns':
        dataset, acc, f1_macro, f1_micro, f1_weighted = test_tfns(model, tokenizer, batch_size = fed_args.test_batch_size)
        Acc, F1_macro, F1_micro, F1_weighted=[], [], [], []
        Acc.append(acc)
        F1_macro.append(f1_macro)
        F1_micro.append(f1_micro)
        F1_weighted.append(f1_weighted)
        np.save(os.path.join(script_args.output_dir, "testing_acc.npy"), np.array(Acc))
        np.save(os.path.join(script_args.output_dir, "f1_macro.npy"), np.array(F1_macro))
        np.save(os.path.join(script_args.output_dir, "f1_micro.npy"), np.array(F1_micro))
        np.save(os.path.join(script_args.output_dir, "f1_weighted.npy"), np.array(F1_weighted))
    elif fed_args.test_dataset=='nwgi':
        dataset, acc, f1_macro, f1_micro, f1_weighted = test_nwgi(model, tokenizer, batch_size = fed_args.test_batch_size)
        Acc, F1_macro, F1_micro, F1_weighted=[], [], [], []
        Acc.append(acc)
        F1_macro.append(f1_macro)
        F1_micro.append(f1_micro)
        F1_weighted.append(f1_weighted)
        np.save(os.path.join(script_args.output_dir, "testing_acc.npy"), np.array(Acc))
        np.save(os.path.join(script_args.output_dir, "f1_macro.npy"), np.array(F1_macro))
        np.save(os.path.join(script_args.output_dir, "f1_micro.npy"), np.array(F1_micro))
        np.save(os.path.join(script_args.output_dir, "f1_weighted.npy"), np.array(F1_weighted))
    else:
        raise ValueError("The test dataset is not supported yet !!!")

