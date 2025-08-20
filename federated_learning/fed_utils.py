import torch
import copy 

def get_proxy_dict(fed_args, global_dict):
    opt_proxy_dict = None
    proxy_dict = None
    if fed_args.fed_alg in ['fedadagrad', 'fedyogi', 'fedadam']:
        proxy_dict, opt_proxy_dict = {}, {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
            opt_proxy_dict[key] = torch.ones_like(global_dict[key]) * fed_args.fedopt_tau**2
    elif fed_args.fed_alg == 'fedavgm':
        proxy_dict = {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])

    return proxy_dict, opt_proxy_dict

def get_auxiliary_dict(fed_args, global_dict):

    if fed_args.fed_alg in ['scaffold']:
        global_auxiliary = {}               # c in SCAFFOLD
        for key in global_dict.keys():
            global_auxiliary[key] = torch.zeros_like(global_dict[key])
        auxiliary_model_list = [copy.deepcopy(global_auxiliary) for _ in range(fed_args.num_clients)]    # c_i in SCAFFOLD
        auxiliary_delta_dict = [copy.deepcopy(global_auxiliary) for _ in range(fed_args.num_clients)]    # delta c_i in SCAFFOLD

    else:
        global_auxiliary = None
        auxiliary_model_list = [None]*fed_args.num_clients
        auxiliary_delta_dict = [None]*fed_args.num_clients

    return global_auxiliary, auxiliary_model_list, auxiliary_delta_dict



def get_proxy_Proj(fed_args, global_dict):
    proxy_Proj={}
    r=fed_args.subspace_rank
    # 注意，lora的层是AB交替的
    flag=0
    for key in global_dict.keys():
        if flag == 0:
            A=torch.zeros_like(global_dict[key])[:,:r]
        proxy_Proj[key]=A
        # print("shape=",proxy_Proj[key].shape)
        # input()
        flag+=1

    return proxy_Proj



def get_client_proxy_Proj(fed_args, global_dict, local_dict_list):
    proxy_Proj={i: {j: None for j in range(128)} for i in range(fed_args.num_clients)}
    for client in range(fed_args.num_clients):
        for key in global_dict.keys():
            if global_dict[key].shape[0] <= global_dict[key].shape[1]:
                proxy_Proj[client][key]=torch.eye(global_dict[key].shape[0])[:,:fed_args.subspace_rank]
            else:
                proxy_Proj[client][key]=torch.eye(global_dict[key].shape[1])[:,:fed_args.subspace_rank]

    return proxy_Proj



def get_sub_proxy_dict(fed_args, global_dict,subspace_rank):
    r=subspace_rank
    sub_opt_proxy_dict = None
    sub_proxy_dict = None
    lora_flag=0
    if fed_args.fed_alg in ['fedadagrad', 'fedyogi', 'fedadam']:
        sub_proxy_dict, sub_opt_proxy_dict = {}, {}
        for key in global_dict.keys():
            if lora_flag % 2 == 0:
                sub_proxy_dict[key] = torch.zeros_like(global_dict[key])[:r,:]
                sub_opt_proxy_dict[key] = torch.ones_like(global_dict[key])[:r,:] * fed_args.fedopt_tau**2
            else:
                sub_proxy_dict[key] = torch.zeros_like(global_dict[key])[:,:r]
                sub_opt_proxy_dict[key] = torch.ones_like(global_dict[key])[:,:r] * fed_args.fedopt_tau**2
            lora_flag+=1
    elif fed_args.fed_alg == 'fedavgm':
        sub_proxy_dict = {}
        for key in global_dict.keys():
            if lora_flag % 2 == 0:
                sub_proxy_dict[key] = torch.zeros_like(global_dict[key])[:r,:]
            else:
                sub_proxy_dict[key] = torch.zeros_like(global_dict[key])[:,:r]
            lora_flag+=1
    # for key in global_dict.keys():
    #     print("sub_proxy_dict shape=",sub_proxy_dict[key].shape)
    #     print("sub_opt_proxy_dict shape=",sub_opt_proxy_dict[key].shape)
    #     input()
    return sub_proxy_dict, sub_opt_proxy_dict

