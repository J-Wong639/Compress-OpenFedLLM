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


from federated_learning.fed_utils import get_proxy_Proj, get_sub_proxy_dict, get_client_proxy_Proj


from datasets import load_metric



import pandas as pd



from modelscope import snapshot_download





def calculate_total_words_alpaca(client_dataset):
    total_words=0
    for sample in client_dataset:
        total_words+=len(sample['instruction'].split())
        total_words+=len(str(sample['response']).split())
    return total_words

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)



dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)

dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample) 



# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)


# ===== Store the words number =====
local_datasets_words_num=list(range(fed_args.num_clients))

for i in range(fed_args.num_clients):
    local_datasets_words_num[i]=calculate_total_words_alpaca(local_datasets[i])


sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config ===== 
device_map, quantization_config, torch_dtype = get_model_config(script_args)



if script_args.local_model_name_or_path is not None:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.local_model_name_or_path,
        quantization_config=quantization_config, #stop pre-train quantization
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config, #stop pre-train quantization
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )




if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
# stop pre-train quantization

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() #在训练阶段会冻结某些参数，只训练部分参数




model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

proxy_Proj=get_client_proxy_Proj(fed_args, global_dict, local_dict_list)
sub_proxy_dict, sub_opt_proxy_dict=get_sub_proxy_dict(fed_args, global_dict, fed_args.subspace_rank)
server_proxy_Proj=get_client_proxy_Proj(fed_args, global_dict, local_dict_list)











# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="left")#用远程模型时 
# tokenizer = AutoTokenizer.from_pretrained(script_args.local_model_name_or_path, use_fast=False, padding_side="right") #用本地的模型时
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer(response_template, add_special_tokens=False)  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
# data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

import pdb; pdb.set_trace()




# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
training_accuracy = [[] for i in range(fed_args.num_clients)]


Acc, F1_macro, F1_micro, F1_weighted=[], [], [], []

save_subrank=[]
save_gradient_norm=[]


save_global_dict_rank=[]


pre_client_delta_w={i: {j: None for j in range(128)} for i in range(fed_args.num_clients)}
cos_sim={i: {j: None for j in range(128)} for i in range(fed_args.num_clients)}



# 记录某个client的某个key有没有被选中过
total_clients=[i for i in range(fed_args.num_clients)]
client_init=copy.deepcopy(proxy_Proj)
if fed_args.fed_alg in ['fedadagrad', 'fedyogi', 'fedadam']:
    for key, param in sub_opt_proxy_dict.items():
        for client in total_clients:
            client_init[client][key]=0
            # print(client_init[client][key])
elif fed_args.fed_alg == 'fedavgm':
    for key, param in sub_proxy_dict.items():
        for client in total_clients:
            client_init[client][key]=0
            # print(client_init[client][key])




for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round)

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        # new_lr=1e-4
        training_args = get_training_args(script_args, new_lr)


        # ===== Train local model on the client side =====
        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
            # eval_dataset=sub_dataset,
            # compute_metrics=compute_metrics,
        )

        results = trainer.train()
        training_loss[client].append(results.training_loss)

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!


    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary, save_subrank, save_gradient_norm, pre_client_delta_w, server_proxy_Proj, client_init = global_aggregate(
        fed_args, script_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round_idx=round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict), \
        global_round=round, \
        proxy_Proj=proxy_Proj, \
        sub_proxy_dict=sub_proxy_dict, \
        sub_opt_proxy_dict=sub_opt_proxy_dict, \
        save_subrank=save_subrank, \
        save_gradient_norm=save_gradient_norm, \
        pre_client_delta_w=pre_client_delta_w, \
        cos_sim=cos_sim, \
        server_proxy_Proj=server_proxy_Proj, \
        client_init=client_init
    )
    set_peft_model_state_dict(model, global_dict)   # Update global model





    # ===== Save the model =====
    if (round+1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
        lora_path=os.path.join(script_args.output_dir, f"checkpoint-{round+1}")
        full_path=os.path.join(script_args.output_dir, f"full-{round+1}")
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))





























