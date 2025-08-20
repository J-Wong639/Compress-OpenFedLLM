import sys
import copy
import os
from tqdm import tqdm
import numpy as np
from typing import Dict
import torch

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, DPOTrainer
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, AutoPeftModelForCausalLM

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args



from federated_learning.fed_utils import get_proxy_Proj, get_sub_proxy_dict, get_client_proxy_Proj

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
dataset = process_dpo_dataset(script_args.dataset_name, dataset, script_args.template, script_args.dataset_sample)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

if script_args.use_peft == True:
    model_ref = None
else:
    # construct a reference model with the identical original parameters
    # e.g. DPO need a reference model to compute the discrepancy loss
    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)


# 原来的方案
# proxy_Proj=get_proxy_Proj(fed_args, global_dict) 
# client_PQ的方案
proxy_Proj=get_client_proxy_Proj(fed_args, global_dict, local_dict_list)
sub_proxy_dict, sub_opt_proxy_dict=get_sub_proxy_dict(fed_args, global_dict, fed_args.subspace_rank)
server_proxy_Proj=get_client_proxy_Proj(fed_args, global_dict, local_dict_list)


# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]


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
            training_loss[client].append(-1)        # -1 is an indicator of not training
            continue

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
        # new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-5)      # manually schedule the learning rate
        new_lr=1e-4
        training_args = get_training_args(script_args, new_lr)

        # ===== Train local model on the client side =====
        trainer = get_fed_local_dpo_trainer(
            script_args, fed_args, model, model_ref, \
            tokenizer, training_args, sub_dataset, global_dict, \
            auxiliary_model_list[client], global_auxiliary
            )
        
        results = trainer.train()
        training_loss[client].append(results.training_loss)

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

    # ===== Aggregate the local models =====
    global_dict, global_auxiliary, save_subrank, save_gradient_norm, pre_client_delta_w, server_proxy_Proj, client_init = global_aggregate(
        fed_args=fed_args, script_args=script_args, global_dict=global_dict, local_dict_list=local_dict_list, sample_num_list=sample_num_list, \
        clients_this_round=clients_this_round, round_idx=round, \
        proxy_dict=proxy_dict, \
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
    set_peft_model_state_dict(model, global_dict)   # update global model

    # ===== Save the model =====
    if (round+1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))