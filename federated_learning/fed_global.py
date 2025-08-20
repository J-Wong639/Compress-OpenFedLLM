import random
import torch
from compression.compression import *
import numpy as np
import pandas as pd
import os
import copy



def get_clients_this_round(fed_args, round):
    if (fed_args.fed_alg).startswith('local'):
        clients_this_round = [int((fed_args.fed_alg)[-1])]
    else:
        if fed_args.num_clients < fed_args.sample_clients:
            clients_this_round = list(range(fed_args.num_clients))
        else:
            random.seed(round)
            clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
    return clients_this_round


def change_gradient_subspace_projection(fed_args, delta_w):
    r=fed_args.subspace_rank
    if delta_w.is_cuda:
        delta_w=delta_w.cpu()
    np_delta_w=delta_w.detach().numpy()
    U,S,VT=np.linalg.svd(np_delta_w)
    # rank = np.sum(S > 1e-10)
    # print("Rank of the matrix:", rank)
    # input()
    #这里和lora的架构有关，因为每次读进来的矩阵是A,B交替的
    if delta_w.shape[0] <= delta_w.shape[1]: 
        P=U[:,:r]
        # print("shape P=",P.shape)
        np_delta_w=np.dot(P.T,np_delta_w)
        np_Proj=P
    else:
        Q=VT.T[:,:r]
        # print("shape Q=",Q.shape)
        np_delta_w=np.dot(np_delta_w,Q)
        np_Proj=Q
    delta_w=torch.from_numpy(np_delta_w)
    delta_w=delta_w.to(device='cuda:0')
    Proj=torch.from_numpy(np_Proj)
    Proj=Proj.to(device='cuda:0')
    if fed_args.is_quantized:
        compressor=QSGDCompressor()
        Proj=compressor.compress(Proj,quantize_level=fed_args.quantize_level)
    return delta_w, Proj  

def gradient_subspace_projection(fed_args, delta_w, Proj):
    if delta_w.is_cuda:
        delta_w=delta_w.cpu()
    if Proj.is_cuda:
        Proj=Proj.cpu()
    np_delta_w=delta_w.detach().numpy()
    np_Proj=Proj.detach().numpy()
    #这里和lora的架构有关，因为每次读进来的矩阵是A,B交替的
    if delta_w.shape[0] <= delta_w.shape[1]: 
        np_delta_w=np.dot(np_Proj.T,np_delta_w)
    else:
        np_delta_w=np.dot(np_delta_w,np_Proj)
    delta_w=torch.from_numpy(np_delta_w)
    delta_w=delta_w.to(device='cuda:0')
    return delta_w


def global_aggregate(fed_args, script_args,  global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None, global_round=0, proxy_Proj=None, sub_proxy_dict=None, sub_opt_proxy_dict=None, save_subrank=None, save_gradient_norm=None, save_client_delta=None, pre_client_delta_w=None, cos_sim=None, server_proxy_Proj=None, client_init=None):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None


    if fed_args.is_quantized:
        compressor=QSGDCompressor()
    elif fed_args.is_topk:
        compressor=TopKCompressor()


    if fed_args.fed_alg == 'scaffold':
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        for key in global_auxiliary.keys():
            delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round]) 
            global_auxiliary[key] += delta_auxiliary / fed_args.num_clients
    
    elif fed_args.fed_alg == 'fedavgm':
        if fed_args.is_change_subspace:
            #初始化dict一定要用这个公式，不要直接赋值local_dict_list，不然会出现很多乱七八糟的错误
            client_delta_w={i: {j: None for j in range(128)} for i in range(fed_args.num_clients)}
            key_flag=0
            for key in global_dict.keys():
                for client in clients_this_round:
                    client_delta_w[client][key]=copy.deepcopy(local_dict_list[client][key]-global_dict[key])
                    if global_round % fed_args.subspace_change_freq == 0:
                        client_delta_w[client][key], proxy_Proj[client][key] = change_gradient_subspace_projection(fed_args, client_delta_w[client][key])
                    else:
                        client_delta_w[client][key] = gradient_subspace_projection(fed_args, client_delta_w[client][key], proxy_Proj[client][key])

                #client_PQ
                    alpha=1
                    state=client_delta_w[client][key].cpu()
                    proxy_Proj[client][key]=proxy_Proj[client][key].cpu()
                    np_state=state.detach().numpy()
                    proxy_Proj[client][key]=proxy_Proj[client][key].detach().numpy()
                    if np_state.shape[0] <= np_state.shape[1]:
                        np_G=alpha*np.dot(proxy_Proj[client][key],np_state)
                    else:
                        np_G=alpha*np.dot(np_state,proxy_Proj[client][key].T)
                    G=torch.from_numpy(np_G)
                    G=G.to(device='cuda:0')
                    client_delta_w[client][key]=G
                    proxy_Proj[client][key]=torch.from_numpy(proxy_Proj[client][key])
                    proxy_Proj[client][key]=proxy_Proj[client][key].to(device='cuda:0')

                delta_w = sum([client_delta_w[client][key] for client in clients_this_round]) / len(clients_this_round)

                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                global_dict[key] = global_dict[key] + proxy_dict[key] #这里learning rate只能设置为1，不然会崩

        else:
            # Momentum-based FedAvg
            for key in global_dict.keys():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
                if fed_args.is_quantized:
                    delta_w=compressor.compress(delta_w,quantize_level=fed_args.quantize_level)
                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                global_dict[key] = global_dict[key] + proxy_dict[key] #这里learning rate只能设置为1，不然会崩

    elif fed_args.fed_alg == 'fedadagrad':
        if fed_args.is_change_subspace:
            #初始化dict一定要用这个公式，不要直接赋值local_dict_list，不然会出现很多乱七八糟的错误
            client_delta_w={i: {j: None for j in range(128)} for i in range(fed_args.num_clients)}
            key_flag=0
            for key, param in sub_opt_proxy_dict.items():
                for client in clients_this_round:
                    client_delta_w[client][key]=copy.deepcopy(local_dict_list[client][key]-global_dict[key])
                    if global_round % fed_args.subspace_change_freq == 0:
                        client_delta_w[client][key], proxy_Proj[client][key] = change_gradient_subspace_projection(fed_args, client_delta_w[client][key])
                    else:
                        client_delta_w[client][key] = gradient_subspace_projection(fed_args, client_delta_w[client][key], proxy_Proj[client][key])

                #client_PQ
                    alpha=1
                    state=client_delta_w[client][key].cpu()
                    proxy_Proj[client][key]=proxy_Proj[client][key].cpu()
                    np_state=state.detach().numpy()
                    proxy_Proj[client][key]=proxy_Proj[client][key].detach().numpy()
                    if np_state.shape[0] <= np_state.shape[1]:
                        np_G=alpha*np.dot(proxy_Proj[client][key],np_state)
                    else:
                        np_G=alpha*np.dot(np_state,proxy_Proj[client][key].T)
                    G=torch.from_numpy(np_G)
                    G=G.to(device='cuda:0')
                    client_delta_w[client][key]=G
                    proxy_Proj[client][key]=torch.from_numpy(proxy_Proj[client][key])
                    proxy_Proj[client][key]=proxy_Proj[client][key].to(device='cuda:0')

                delta_w = sum([client_delta_w[client][key] for client in clients_this_round]) / len(clients_this_round)
                
                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                opt_proxy_dict[key] = opt_proxy_dict[key] + torch.square(delta_w)
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

        else:
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                if fed_args.is_quantized:
                    delta_w=compressor.compress(delta_w,quantize_level=fed_args.quantize_level)
                # In paper 'adaptive federated optimization', momentum is not used
                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                opt_proxy_dict[key] = opt_proxy_dict[key] + torch.square(delta_w)
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)


    #clientPQ
    elif fed_args.fed_alg == 'fedyogi':
        if fed_args.is_change_subspace:
            #初始化dict一定要用这个公式，不要直接赋值local_dict_list，不然会出现很多乱七八糟的错误
            client_delta_w={i: {j: None for j in range(128)} for i in range(fed_args.num_clients)}
            key_flag=0
            for key, param in sub_opt_proxy_dict.items():
                for client in clients_this_round:
                    client_delta_w[client][key]=copy.deepcopy(local_dict_list[client][key]-global_dict[key])
                    if global_round % fed_args.subspace_change_freq == 0:
                        client_delta_w[client][key], proxy_Proj[client][key] = change_gradient_subspace_projection(fed_args, client_delta_w[client][key])
                    else:
                        client_delta_w[client][key] = gradient_subspace_projection(fed_args, client_delta_w[client][key], proxy_Proj[client][key])
                #client_PQ
                    alpha=1
                    state=client_delta_w[client][key].cpu()
                    proxy_Proj[client][key]=proxy_Proj[client][key].cpu()
                    np_state=state.detach().numpy()
                    proxy_Proj[client][key]=proxy_Proj[client][key].detach().numpy()
                    if np_state.shape[0] <= np_state.shape[1]:
                        np_G=alpha*np.dot(proxy_Proj[client][key],np_state)
                    else:
                        np_G=alpha*np.dot(np_state,proxy_Proj[client][key].T)
                    G=torch.from_numpy(np_G)
                    G=G.to(device='cuda:0')
                    client_delta_w[client][key]=G
                    proxy_Proj[client][key]=torch.from_numpy(proxy_Proj[client][key])
                    proxy_Proj[client][key]=proxy_Proj[client][key].to(device='cuda:0')
                
                delta_w = sum([client_delta_w[client][key] for client in clients_this_round]) / len(clients_this_round)
                    
                # global aggregation
                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                delta_square = torch.square(delta_w)
                opt_proxy_dict[key] = opt_proxy_dict[key] - (1-fed_args.fedopt_beta2)*delta_square*torch.sign(opt_proxy_dict[key] - delta_square)
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)
                    # # reverse projection
                    # alpha=1
                    # N=N.cpu()
                    # proxy_Proj[key]=proxy_Proj[key].cpu()
                    # np_N=N.detach().numpy()
                    # proxy_Proj[key]=proxy_Proj[key].detach().numpy()
                    # if np_N.shape[0] <= np_N.shape[1]:
                    #     np_G=alpha*np.dot(proxy_Proj[key],np_N)
                    # else:
                    #     np_G=alpha*np.dot(np_N,proxy_Proj[key].T)
                    # G=torch.from_numpy(np_G)
                    # G=G.to(device='cuda:0')
                    # proxy_Proj[key]=torch.from_numpy(proxy_Proj[key])
                    # proxy_Proj[key]=proxy_Proj[key].to(device='cuda:0')
                    # # global update
                    # global_dict[key] += fed_args.fedopt_eta * G
        else:
            if fed_args.is_topk:
                compressor.clear()
                compressed_paras={}
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                if fed_args.is_quantized:
                    delta_w=compressor.compress(delta_w,quantize_level=fed_args.quantize_level)
                elif fed_args.is_topk:
                    # compress
                    # fed_argscompression_ratio=0.1 # 意思是只保留0.1的参数，其他都置为0了
                    flatten_tensor=compressor.flatten(delta_w,key)
                    compressor.compress(flatten_tensor, name=key, sigma_scale=2.5, ratio=fed_args.topk_compression_ratio)
                    compressed_paras[key]=compressor.values[key]
                    # decompress
                    decompress_tensor=compressor.decompress_new(compressed_paras[key], compressor.indexes[key], name=key, shape=compressor.shapes[key])
                    delta_w=compressor.unflatten(decompress_tensor,name=key,shape=None)


                    


                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                delta_square = torch.square(delta_w)
                opt_proxy_dict[key] = opt_proxy_dict[key] - (1-fed_args.fedopt_beta2)*delta_square*torch.sign(opt_proxy_dict[key] - delta_square)
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)





    #serverPQ
    elif fed_args.fed_alg == 'fedyogi_serverPQ':
        if fed_args.is_change_subspace:
            for key, param in sub_opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                if fed_args.is_quantized:
                    delta_w=compressor.compress(delta_w,quantize_level=fed_args.quantize_level)
                # 在global_round==0的时候会进入初始化
                # print("delta_w shape=", delta_w.shape)
                # print("proxy_Proj shape=", proxy_Proj[key].shape)
                if global_round % fed_args.subspace_change_freq == 0:
                    delta_w, proxy_Proj[key] = change_gradient_subspace_projection(fed_args,  delta_w)
                    # print(proxy_Proj[key].shape) # 全是(32,16)
                    # print(delta_w.shape) # (16,4096) (4096,16) 这样交替出现
                else:
                    delta_w = gradient_subspace_projection(fed_args,  delta_w, proxy_Proj[key])
 
                # global aggregation
                sub_proxy_dict[key] = fed_args.fedopt_beta1 * sub_proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                delta_square = torch.square(delta_w)
                sub_opt_proxy_dict[key] = sub_opt_proxy_dict[key] - (1-fed_args.fedopt_beta2)*delta_square*torch.sign(sub_opt_proxy_dict[key] - delta_square)
                N = fed_args.fedopt_eta * torch.div(sub_proxy_dict[key], torch.sqrt(sub_opt_proxy_dict[key])+fed_args.fedopt_tau)
               



                # reverse projection
                alpha=1
                N=N.cpu()
                proxy_Proj[key]=proxy_Proj[key].cpu()
                np_N=N.detach().numpy()
                proxy_Proj[key]=proxy_Proj[key].detach().numpy()
                if np_N.shape[0] <= np_N.shape[1]:
                    np_G=alpha*np.dot(proxy_Proj[key],np_N)
                else:
                    np_G=alpha*np.dot(np_N,proxy_Proj[key].T)
                G=torch.from_numpy(np_G)
                G=G.to(device='cuda:0')
                proxy_Proj[key]=torch.from_numpy(proxy_Proj[key])
                proxy_Proj[key]=proxy_Proj[key].to(device='cuda:0')
                # global update
                global_dict[key] += fed_args.fedopt_eta * G




                

        else:
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                if fed_args.is_quantized:
                    delta_w=compressor.compress(delta_w,quantize_level=fed_args.quantize_level)
                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                delta_square = torch.square(delta_w)
                opt_proxy_dict[key] = opt_proxy_dict[key] - (1-fed_args.fedopt_beta2)*delta_square*torch.sign(opt_proxy_dict[key] - delta_square)
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)





    # client用各自的PQ压缩,server用approx的PQ, server每隔r轮更新一次PQ
    elif fed_args.fed_alg == 'fedadam_save':
        if fed_args.is_change_subspace:
            #初始化dict一定要用这个公式，不要直接赋值local_dict_list，不然会出现很多乱七八糟的错误
            client_delta_w={i: {j: None for j in range(128)} for i in range(fed_args.num_clients)}
            key_flag=0
            for key, param in sub_opt_proxy_dict.items():
                for client in clients_this_round:
                    client_delta_w[client][key]=copy.deepcopy(local_dict_list[client][key]-global_dict[key])
                    # client压缩
                    if client_init[client][key] == 0:
                        # total_clients=[i for i in range(fed_args.num_clients)]
                        client_delta_w[client][key], proxy_Proj[client][key] = change_gradient_subspace_projection(fed_args, client_delta_w[client][key])
                        server_proxy_Proj[client][key]=copy.deepcopy(proxy_Proj[client][key])
                        client_init[client][key]=1
                        # server解压缩
                        alpha=1
                        state=client_delta_w[client][key].cpu()
                        proxy_Proj[client][key]=proxy_Proj[client][key].cpu()
                        np_state=state.detach().numpy()
                        proxy_Proj[client][key]=proxy_Proj[client][key].detach().numpy()
                        if np_state.shape[0] <= np_state.shape[1]:
                            np_G=alpha*np.dot(proxy_Proj[client][key],np_state)
                        else:
                            np_G=alpha*np.dot(np_state,proxy_Proj[client][key].T)
                        G=torch.from_numpy(np_G)
                        G=G.to(device='cuda:0')
                        client_delta_w[client][key]=G
                        proxy_Proj[client][key]=torch.from_numpy(proxy_Proj[client][key])
                        proxy_Proj[client][key]=proxy_Proj[client][key].to(device='cuda:0')

                        pre_client_delta_w[client][key]=copy.deepcopy(client_delta_w[client][key])
                    elif global_round % fed_args.subspace_change_freq == 0:
                        # total_clients=[i for i in range(fed_args.num_clients)]
                        client_delta_w[client][key], proxy_Proj[client][key] = change_gradient_subspace_projection(fed_args, client_delta_w[client][key])
                        server_proxy_Proj[client][key]=copy.deepcopy(proxy_Proj[client][key])
                        # server解压缩
                        alpha=1
                        state=client_delta_w[client][key].cpu()
                        proxy_Proj[client][key]=proxy_Proj[client][key].cpu()
                        np_state=state.detach().numpy()
                        proxy_Proj[client][key]=proxy_Proj[client][key].detach().numpy()
                        if np_state.shape[0] <= np_state.shape[1]:
                            np_G=alpha*np.dot(proxy_Proj[client][key],np_state)
                        else:
                            np_G=alpha*np.dot(np_state,proxy_Proj[client][key].T)
                        G=torch.from_numpy(np_G)
                        G=G.to(device='cuda:0')
                        client_delta_w[client][key]=G
                        proxy_Proj[client][key]=torch.from_numpy(proxy_Proj[client][key])
                        proxy_Proj[client][key]=proxy_Proj[client][key].to(device='cuda:0')

                        pre_client_delta_w[client][key]=copy.deepcopy(client_delta_w[client][key])
                    else:
                        # client投影
                        client_delta_w[client][key] = gradient_subspace_projection(fed_args, client_delta_w[client][key], proxy_Proj[client][key])

                        # 生成server投影矩阵
                        R=client_delta_w[client][key].cpu()
                        np_R=R.detach().numpy()

                        pre_G=pre_client_delta_w[client][key].cpu()
                        np_pre_G=pre_G.detach().numpy()

                        np_R_pinv=np.linalg.pinv(np_R)

                        if np_R.shape[0] <= np_R.shape[1]:
                            np_server_Proj=np.dot(np_pre_G,np_R_pinv)
                        else:
                            np_server_Proj=(np.dot(np_R_pinv,np_pre_G)).T

                        server_Proj=torch.from_numpy(np_server_Proj)
                        server_Proj=server_Proj.to(device='cuda:0')

                        server_proxy_Proj[client][key]=server_Proj

                        R=torch.from_numpy(np_R)
                        R=R.to(device='cuda:0')
                        pre_G=torch.from_numpy(np_pre_G)
                        pre_G=pre_G.to(device='cuda:0')

                        # server重投影
                        alpha=1
                        state=client_delta_w[client][key].cpu()
                        server_proxy_Proj[client][key]=server_proxy_Proj[client][key].cpu()
                        np_state=state.detach().numpy()
                        server_proxy_Proj[client][key]=server_proxy_Proj[client][key].detach().numpy()
                        if np_state.shape[0] <= np_state.shape[1]:
                            np_G=alpha*np.dot(server_proxy_Proj[client][key],np_state)
                        else:
                            np_G=alpha*np.dot(np_state,server_proxy_Proj[client][key].T)
                        G=torch.from_numpy(np_G)
                        G=G.to(device='cuda:0')
                        client_delta_w[client][key]=G


                        server_proxy_Proj[client][key]=torch.from_numpy(server_proxy_Proj[client][key])
                        server_proxy_Proj[client][key]=server_proxy_Proj[client][key].to(device='cuda:0')
                        state=torch.from_numpy(np_state)
                        state=state.to(device='cuda:0')

                        pre_client_delta_w[client][key]=client_delta_w[client][key]
 
                delta_w = sum([client_delta_w[client][key] for client in clients_this_round]) / len(clients_this_round)


                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                opt_proxy_dict[key] = fed_args.fedopt_beta2*opt_proxy_dict[key] + (1-fed_args.fedopt_beta2)*torch.square(delta_w)
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)  
                key_flag+=1
        else:
            key_flag=0
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                if fed_args.is_quantized:
                    delta_w=compressor.compress(delta_w,quantize_level=fed_args.quantize_level)

                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                opt_proxy_dict[key] = fed_args.fedopt_beta2*opt_proxy_dict[key] + (1-fed_args.fedopt_beta2)*torch.square(delta_w)
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)
                key_flag+=1

    # client用各自的PQ压缩,server用approx的PQ
    elif fed_args.fed_alg == 'fedadam_save':
        if fed_args.is_change_subspace:
            #初始化dict一定要用这个公式，不要直接赋值local_dict_list，不然会出现很多乱七八糟的错误
            client_delta_w={i: {j: None for j in range(128)} for i in range(fed_args.num_clients)}
            key_flag=0
            for key, param in sub_opt_proxy_dict.items():
                for client in clients_this_round:
                    client_delta_w[client][key]=copy.deepcopy(local_dict_list[client][key]-global_dict[key])
                    # client压缩
                    if client_init[client][key] == 0:
                        # total_clients=[i for i in range(fed_args.num_clients)]
                        client_delta_w[client][key], proxy_Proj[client][key] = change_gradient_subspace_projection(fed_args, client_delta_w[client][key])
                        server_proxy_Proj[client][key]=copy.deepcopy(proxy_Proj[client][key])
                        client_init[client][key]=1
                        # server解压缩
                        alpha=1
                        state=client_delta_w[client][key].cpu()
                        proxy_Proj[client][key]=proxy_Proj[client][key].cpu()
                        np_state=state.detach().numpy()
                        proxy_Proj[client][key]=proxy_Proj[client][key].detach().numpy()
                        if np_state.shape[0] <= np_state.shape[1]:
                            np_G=alpha*np.dot(proxy_Proj[client][key],np_state)
                        else:
                            np_G=alpha*np.dot(np_state,proxy_Proj[client][key].T)
                        G=torch.from_numpy(np_G)
                        G=G.to(device='cuda:0')
                        client_delta_w[client][key]=G
                        proxy_Proj[client][key]=torch.from_numpy(proxy_Proj[client][key])
                        proxy_Proj[client][key]=proxy_Proj[client][key].to(device='cuda:0')

                        pre_client_delta_w[client][key]=copy.deepcopy(client_delta_w[client][key])
                    elif global_round % fed_args.subspace_change_freq == 0:
                        # client投影
                        client_delta_w[client][key], proxy_Proj[client][key] = change_gradient_subspace_projection(fed_args, client_delta_w[client][key])

                        # 生成server投影矩阵
                        R=client_delta_w[client][key].cpu()
                        np_R=R.detach().numpy()

                        pre_G=pre_client_delta_w[client][key].cpu()
                        np_pre_G=pre_G.detach().numpy()

                        np_R_pinv=np.linalg.pinv(np_R)

                        if np_R.shape[0] <= np_R.shape[1]:
                            np_server_Proj=np.dot(np_pre_G,np_R_pinv)
                        else:
                            np_server_Proj=(np.dot(np_R_pinv,np_pre_G)).T

                        server_Proj=torch.from_numpy(np_server_Proj)
                        server_Proj=server_Proj.to(device='cuda:0')

                        server_proxy_Proj[client][key]=server_Proj

                        R=torch.from_numpy(np_R)
                        R=R.to(device='cuda:0')
                        pre_G=torch.from_numpy(np_pre_G)
                        pre_G=pre_G.to(device='cuda:0')

                        # server重投影
                        alpha=1
                        state=client_delta_w[client][key].cpu()
                        server_proxy_Proj[client][key]=server_proxy_Proj[client][key].cpu()
                        np_state=state.detach().numpy()
                        server_proxy_Proj[client][key]=server_proxy_Proj[client][key].detach().numpy()
                        if np_state.shape[0] <= np_state.shape[1]:
                            np_G=alpha*np.dot(server_proxy_Proj[client][key],np_state)
                        else:
                            np_G=alpha*np.dot(np_state,server_proxy_Proj[client][key].T)
                        G=torch.from_numpy(np_G)
                        G=G.to(device='cuda:0')
                        client_delta_w[client][key]=G


                        server_proxy_Proj[client][key]=torch.from_numpy(server_proxy_Proj[client][key])
                        server_proxy_Proj[client][key]=server_proxy_Proj[client][key].to(device='cuda:0')
                        state=torch.from_numpy(np_state)
                        state=state.to(device='cuda:0')

                        pre_client_delta_w[client][key]=client_delta_w[client][key]
                    else:
                        # client投影
                        client_delta_w[client][key] = gradient_subspace_projection(fed_args, client_delta_w[client][key], proxy_Proj[client][key])

                        # 生成server投影矩阵
                        R=client_delta_w[client][key].cpu()
                        np_R=R.detach().numpy()

                        pre_G=pre_client_delta_w[client][key].cpu()
                        np_pre_G=pre_G.detach().numpy()

                        np_R_pinv=np.linalg.pinv(np_R)

                        if np_R.shape[0] <= np_R.shape[1]:
                            np_server_Proj=np.dot(np_pre_G,np_R_pinv)
                        else:
                            np_server_Proj=(np.dot(np_R_pinv,np_pre_G)).T

                        server_Proj=torch.from_numpy(np_server_Proj)
                        server_Proj=server_Proj.to(device='cuda:0')

                        server_proxy_Proj[client][key]=server_Proj

                        R=torch.from_numpy(np_R)
                        R=R.to(device='cuda:0')
                        pre_G=torch.from_numpy(np_pre_G)
                        pre_G=pre_G.to(device='cuda:0')

                        # server重投影
                        alpha=1
                        state=client_delta_w[client][key].cpu()
                        server_proxy_Proj[client][key]=server_proxy_Proj[client][key].cpu()
                        np_state=state.detach().numpy()
                        server_proxy_Proj[client][key]=server_proxy_Proj[client][key].detach().numpy()
                        if np_state.shape[0] <= np_state.shape[1]:
                            np_G=alpha*np.dot(server_proxy_Proj[client][key],np_state)
                        else:
                            np_G=alpha*np.dot(np_state,server_proxy_Proj[client][key].T)
                        G=torch.from_numpy(np_G)
                        G=G.to(device='cuda:0')
                        client_delta_w[client][key]=G


                        server_proxy_Proj[client][key]=torch.from_numpy(server_proxy_Proj[client][key])
                        server_proxy_Proj[client][key]=server_proxy_Proj[client][key].to(device='cuda:0')
                        state=torch.from_numpy(np_state)
                        state=state.to(device='cuda:0')

                        pre_client_delta_w[client][key]=client_delta_w[client][key]
 
                delta_w = sum([client_delta_w[client][key] for client in clients_this_round]) / len(clients_this_round)


                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                opt_proxy_dict[key] = fed_args.fedopt_beta2*opt_proxy_dict[key] + (1-fed_args.fedopt_beta2)*torch.square(delta_w)
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

                # # #取出每个layer每个client的P或Q         
                # if key_flag == 127:
                #     key_array=np.full((1,32),f"layer{key_flag+1}")
                #     for client in clients_this_round:
                #         client_Proj=proxy_Proj[client][key]
                #         client_Proj=client_Proj.T
                #         if client_Proj.is_cuda:
                #             client_Proj=client_Proj.cpu()
                #         np_client_Proj=client_Proj.detach().numpy()
                        
                #         name_array=np.full((1,32),f"client{client}")
                #         client_array=np.concatenate((key_array,name_array), axis=0)
                #         client_array=np.concatenate((client_array, np_client_Proj), axis=0)
                #         if not os.path.exists(os.path.join(script_args.output_dir,f"layer{key_flag+1}")):
                #             os.makedirs(os.path.join(script_args.output_dir,f"layer{key_flag+1}"))
                #         np.save(os.path.join(script_args.output_dir,f"layer{key_flag+1}/local_round{global_round}_client{client}.npy"), client_array)
                #         # print(client_array.shape)
                #         client_Proj=torch.from_numpy(np_client_Proj)
                #         client_Proj=client_Proj.to(device='cuda:0')    

                # # #取出每个layer每个client的server_P或server_Q         
                # if key_flag == 127:
                #     key_array=np.full((1,32),f"layer{key_flag+1}")
                #     for client in clients_this_round:
                #         client_Proj=server_proxy_Proj[client][key]
                #         client_Proj=client_Proj.T
                #         if client_Proj.is_cuda:
                #             client_Proj=client_Proj.cpu()
                #         np_client_Proj=client_Proj.detach().numpy()
                        
                #         name_array=np.full((1,32),f"client{client}")
                #         client_array=np.concatenate((key_array,name_array), axis=0)
                #         client_array=np.concatenate((client_array, np_client_Proj), axis=0)
                #         if not os.path.exists(os.path.join(script_args.output_dir,f"layer{key_flag+1}")):
                #             os.makedirs(os.path.join(script_args.output_dir,f"layer{key_flag+1}"))
                #         np.save(os.path.join(script_args.output_dir,f"layer{key_flag+1}/global_round{global_round}_client{client}.npy"), client_array)
                #         # print(client_array.shape)
                #         client_Proj=torch.from_numpy(np_client_Proj)
                #         client_Proj=client_Proj.to(device='cuda:0')    
                key_flag+=1
        else:
            key_flag=0
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                if fed_args.is_quantized:
                    delta_w=compressor.compress(delta_w,quantize_level=fed_args.quantize_level)

                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                opt_proxy_dict[key] = fed_args.fedopt_beta2*opt_proxy_dict[key] + (1-fed_args.fedopt_beta2)*torch.square(delta_w)
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)
                key_flag+=1

    # clientPQ
    elif fed_args.fed_alg == 'fedadam':
        if fed_args.is_change_subspace:
            #初始化dict一定要用这个公式，不要直接赋值local_dict_list，不然会出现很多乱七八糟的错误
            client_delta_w={i: {j: None for j in range(128)} for i in range(fed_args.num_clients)}
            key_flag=0
            for key, param in sub_opt_proxy_dict.items():
                for client in clients_this_round:
                    client_delta_w[client][key]=copy.deepcopy(local_dict_list[client][key]-global_dict[key])
                    if global_round % fed_args.subspace_change_freq == 0:
                        client_delta_w[client][key], proxy_Proj[client][key] = change_gradient_subspace_projection(fed_args, client_delta_w[client][key])
                    else:
                        client_delta_w[client][key] = gradient_subspace_projection(fed_args, client_delta_w[client][key], proxy_Proj[client][key])

                #client_PQ
                    alpha=1
                    state=client_delta_w[client][key].cpu()
                    proxy_Proj[client][key]=proxy_Proj[client][key].cpu()
                    np_state=state.detach().numpy()
                    proxy_Proj[client][key]=proxy_Proj[client][key].detach().numpy()
                    if np_state.shape[0] <= np_state.shape[1]:
                        np_G=alpha*np.dot(proxy_Proj[client][key],np_state)
                    else:
                        np_G=alpha*np.dot(np_state,proxy_Proj[client][key].T)
                    G=torch.from_numpy(np_G)
                    G=G.to(device='cuda:0')
                    client_delta_w[client][key]=G
                    proxy_Proj[client][key]=torch.from_numpy(proxy_Proj[client][key])
                    proxy_Proj[client][key]=proxy_Proj[client][key].to(device='cuda:0')
                
                delta_w = sum([client_delta_w[client][key] for client in clients_this_round]) / len(clients_this_round)


                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                opt_proxy_dict[key] = fed_args.fedopt_beta2*opt_proxy_dict[key] + (1-fed_args.fedopt_beta2)*torch.square(delta_w)
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)
                #client_PQ

                # #取出每个layer每个client的P或Q         
                # if key_flag % 4 == 0:
                # if key_flag == 127:
                #     key_array=np.full((1,32),f"layer{key_flag+1}")
                #     for client in clients_this_round:
                #         client_Proj=proxy_Proj[client][key]
                #         client_Proj=client_Proj.T
                #         if client_Proj.is_cuda:
                #             client_Proj=client_Proj.cpu()
                #         np_client_Proj=client_Proj.detach().numpy()
                        
                #         name_array=np.full((1,32),f"client{client}")
                #         client_array=np.concatenate((key_array,name_array), axis=0)
                #         client_array=np.concatenate((client_array, np_client_Proj), axis=0)
                #         if not os.path.exists(os.path.join(script_args.output_dir,f"layer{key_flag+1}")):
                #             os.makedirs(os.path.join(script_args.output_dir,f"layer{key_flag+1}"))
                #         np.save(os.path.join(script_args.output_dir,f"layer{key_flag+1}/round{global_round}_client{client}.npy"), client_array)
                #         # print(client_array.shape)
                #         client_Proj=torch.from_numpy(np_client_Proj)
                #         client_Proj=client_Proj.to(device='cuda:0')    
                # key_flag+=1
                # #取出每个layer每个client的P或Q

                # # #global_PQ
                # Proj=None
                # flag=None
                # for client in clients_this_round:
                #     if proxy_Proj[client][key].shape[0] <= proxy_Proj[client][key].shape[1]:
                #         expanded_tensor=torch.zeros(proxy_Proj[client][key].shape[1],proxy_Proj[client][key].shape[1])
                #         expanded_tensor[:proxy_Proj[client][key].shape[0],:]=proxy_Proj[client][key]
                #         flag=0
                #     else:
                #         expanded_tensor=torch.zeros(proxy_Proj[client][key].shape[0],proxy_Proj[client][key].shape[0])
                #         expanded_tensor[:,:proxy_Proj[client][key].shape[1]]=proxy_Proj[client][key]
                #         flag=1
                #     if Proj==None:
                #         Proj=expanded_tensor
                #     else:
                #         Proj+=expanded_tensor
                # Proj=Proj/len(clients_this_round)
                # if flag == 0:
                #     Proj=Proj[:fed_args.subspace_rank,:]
                # elif flag == 1:
                #     Proj=Proj[:,:fed_args.subspace_rank]



                # alpha=1
                # Proj=Proj.cpu()
                # np_Proj=Proj.detach().numpy()
                # for client in clients_this_round:
                #     state=client_delta_w[client][key].cpu()
                #     np_state=state.detach().numpy()
                #     if np_state.shape[0] <= np_state.shape[1]:
                #         np_G=alpha*np.dot(np_Proj,np_state)
                #     else:
                #         np_G=alpha*np.dot(np_state,np_Proj.T)
                #     G=torch.from_numpy(np_G)
                #     G=G.to(device='cuda:0')
                #     client_delta_w[client][key]=G
                # delta_w = sum([client_delta_w[client][key] for client in clients_this_round]) / len(clients_this_round)
                # proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                # opt_proxy_dict[key] = fed_args.fedopt_beta2*opt_proxy_dict[key] + (1-fed_args.fedopt_beta2)*torch.square(delta_w)
                # global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)
                # # #global_PQ
        else:
            if fed_args.is_topk:
                compressor.clear()
                compressed_paras={}
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                if fed_args.is_quantized:
                    delta_w=compressor.compress(delta_w,quantize_level=fed_args.quantize_level)
                elif fed_args.is_topk:
                    # compress
                    # fed_argscompression_ratio=0.1 # 意思是只保留0.1的参数，其他都置为0了
                    flatten_tensor=compressor.flatten(delta_w,key)
                    compressor.compress(flatten_tensor, name=key, sigma_scale=2.5, ratio=fed_args.topk_compression_ratio)
                    compressed_paras[key]=compressor.values[key]
                    # decompress
                    decompress_tensor=compressor.decompress_new(compressed_paras[key], compressor.indexes[key], name=key, shape=compressor.shapes[key])
                    delta_w=compressor.unflatten(decompress_tensor,name=key,shape=None)

                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                opt_proxy_dict[key] = fed_args.fedopt_beta2*opt_proxy_dict[key] + (1-fed_args.fedopt_beta2)*torch.square(delta_w)
                # global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    # serverPQ
    elif fed_args.fed_alg == 'fedadam_serverPQ':
        if fed_args.is_change_subspace:
            for key, param in sub_opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                if fed_args.is_quantized:
                    delta_w=compressor.compress(delta_w,quantize_level=fed_args.quantize_level)
                # 在global_round==0的时候会进入初始化
                # print("delta_w shape=", delta_w.shape)
                # print("proxy_Proj shape=", proxy_Proj[key].shape)
                if global_round % fed_args.subspace_change_freq == 0:
                    delta_w, proxy_Proj[key] = change_gradient_subspace_projection(fed_args,  delta_w)
                    # print(proxy_Proj[key].shape) # 全是(32,16)
                    # print(delta_w.shape) # (16,4096) (4096,16) 这样交替出现
                else:
                    delta_w = gradient_subspace_projection(fed_args,  delta_w, proxy_Proj[key])
                    # print(proxy_Proj[key].shape) # 全是(32,16)
                    # print(delta_w.shape) # (16,4096) (4096,16) 这样交替出现
                # global aggregation
                # print("shape delta_w=",delta_w.shape)
                # print("proxy_Proj shape=", proxy_Proj[key].shape)
                sub_proxy_dict[key] = fed_args.fedopt_beta1 * sub_proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                sub_opt_proxy_dict[key] = fed_args.fedopt_beta2*sub_opt_proxy_dict[key] + (1-fed_args.fedopt_beta2)*torch.square(delta_w)
                N = torch.div(sub_proxy_dict[key], torch.sqrt(sub_opt_proxy_dict[key])+fed_args.fedopt_tau)
                # reverse projection
                alpha=1
                N=N.cpu()
                proxy_Proj[key]=proxy_Proj[key].cpu()
                np_N=N.detach().numpy()
                proxy_Proj[key]=proxy_Proj[key].detach().numpy()
                if np_N.shape[0] <= np_N.shape[1]:
                    np_G=alpha*np.dot(proxy_Proj[key],np_N)
                else:
                    np_G=alpha*np.dot(np_N,proxy_Proj[key].T)
                G=torch.from_numpy(np_G)
                G=G.to(device='cuda:0')
                proxy_Proj[key]=torch.from_numpy(proxy_Proj[key])
                proxy_Proj[key]=proxy_Proj[key].to(device='cuda:0')
                # global update
                global_dict[key] += fed_args.fedopt_eta * G
        else:
            key_flag=0
            for key, param in opt_proxy_dict.items():

                # # #取出每个layer每个client的gradient              
                # # if key_flag % 4 == 0:
                # if key_flag % 128 == 0:
                #     key_array=np.full((1,4096),f"layer{key_flag+1}")
                #     for client in clients_this_round:
                #         client_delta_w=(local_dict_list[client][key] - global_dict[key])
                #         if client_delta_w.is_cuda:
                #             client_delta_w=client_delta_w.cpu()
                #         np_client_delta_w=client_delta_w.detach().numpy()
                        
                #         name_array=np.full((1,4096),f"client{client}")
                #         client_array=np.concatenate((key_array,name_array), axis=0)
                #         client_array=np.concatenate((client_array, np_client_delta_w), axis=0)
                #         if not os.path.exists(os.path.join(script_args.output_dir,f"layer{key_flag+1}")):
                #             os.makedirs(os.path.join(script_args.output_dir,f"layer{key_flag+1}"))
                #         np.save(os.path.join(script_args.output_dir,f"layer{key_flag+1}/round{global_round}_client{client}.npy"), client_array)
                #         # print(client_array.shape)
                #         client_delta_w=torch.from_numpy(np_client_delta_w)
                #         client_delta_w=client_delta_w.to(device='cuda:0')    
                # key_flag+=1
                # # #取出每个layer每个client的gradient

                # # 测试每个client的delta_w的秩
                # for client in clients_this_round:
                #     client_delta_w=(local_dict_list[client][key] - global_dict[key])
                #     if client_delta_w.is_cuda:
                #         client_delta_w=client_delta_w.cpu()
                #     np_client_delta_w=client_delta_w.detach().numpy()
                #     U,S,VT=np.linalg.svd(np_client_delta_w)
                #     S=S.tolist()
                #     name_list=[f"round{global_round}",f"client{client}", f"layer{key}"]
                #     rank_list=name_list+S
                #     print(name_list)
                #     save_subrank.append(rank_list)
                #     client_delta_w=torch.from_numpy(np_client_delta_w)
                #     client_delta_w=client_delta_w.to(device='cuda:0')
                # # 测试每个client的delta_w的秩

                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                if fed_args.is_quantized:
                    delta_w=compressor.compress(delta_w,quantize_level=fed_args.quantize_level)
                
                # # 测试delta_w的秩
                # if delta_w.is_cuda:
                #     delta_w=delta_w.cpu()
                # np_delta_w=delta_w.detach().numpy()
                # norm=np.linalg.norm(np_delta_w)
                # save_gradient_norm.append(norm)
                # U,S,VT=np.linalg.svd(np_delta_w)
                # save_subrank.append(S)
                # delta_w=torch.from_numpy(np_delta_w)
                # delta_w=delta_w.to(device='cuda:0')
                # # 测试delta_w的秩

                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                opt_proxy_dict[key] = fed_args.fedopt_beta2*opt_proxy_dict[key] + (1-fed_args.fedopt_beta2)*torch.square(delta_w)
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)
    # server生成PQ2
    elif fed_args.fed_alg == 'fedadam_save2':
        if fed_args.is_change_subspace:
            key_flag=0
            for key, param in sub_opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                if fed_args.is_quantized:
                    delta_w=compressor.compress(delta_w,quantize_level=fed_args.quantize_level)
                # 在global_round==0的时候会进入初始化
                if global_round == 0:
                    delta_w, proxy_Proj[key] = change_gradient_subspace_projection(fed_args,  delta_w)
                    # print(proxy_Proj[key].shape) # 全是(32,16)
                    # print(delta_w.shape) # (16,4096) (4096,16) 这样交替出现
                else:
                    delta_w = gradient_subspace_projection(fed_args,  delta_w, proxy_Proj[key])
                    # print(proxy_Proj[key].shape) # 全是(32,16)
                    # print(delta_w.shape) # (16,4096) (4096,16) 这样交替出现
                # global aggregation
                # print("shape delta_w=",delta_w.shape)
                # print("proxy_Proj shape=", proxy_Proj[key].shape)
                sub_proxy_dict[key] = fed_args.fedopt_beta1 * sub_proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                sub_opt_proxy_dict[key] = fed_args.fedopt_beta2*sub_opt_proxy_dict[key] + (1-fed_args.fedopt_beta2)*torch.square(delta_w)
                N = torch.div(sub_proxy_dict[key], torch.sqrt(sub_opt_proxy_dict[key])+fed_args.fedopt_tau)
                # reverse projection
                alpha=1
                N=N.cpu()
                proxy_Proj[key]=proxy_Proj[key].cpu()
                np_N=N.detach().numpy()
                proxy_Proj[key]=proxy_Proj[key].detach().numpy()
                if np_N.shape[0] <= np_N.shape[1]:
                    np_G=alpha*np.dot(proxy_Proj[key],np_N)
                else:
                    np_G=alpha*np.dot(np_N,proxy_Proj[key].T)
                G=torch.from_numpy(np_G)
                G=G.to(device='cuda:0')
                proxy_Proj[key]=torch.from_numpy(proxy_Proj[key])
                proxy_Proj[key]=proxy_Proj[key].to(device='cuda:0')
                # global update
                global_dict[key] += fed_args.fedopt_eta * G
                if global_round % fed_args.subspace_change_freq == 0:
                    G, proxy_Proj[key] = change_gradient_subspace_projection(fed_args, G)
                else:
                    pass
                # #取出每个layer的server的P或Q         
                # if key_flag % 4 == 0:
                if key_flag==127:
                    key_array=np.full((1,32),f"layer{key_flag+1}")
                    server_Proj=proxy_Proj[key]
                    server_Proj=server_Proj.T
                    if server_Proj.is_cuda:
                        server_Proj=server_Proj.cpu()
                    np_server_Proj=server_Proj.detach().numpy()
                    name_array=np.full((1,32),f"server")
                    server_array=np.concatenate((key_array,name_array), axis=0)
                    server_array=np.concatenate((server_array, np_server_Proj), axis=0)
                    if not os.path.exists(os.path.join(script_args.output_dir,f"layer{key_flag+1}")):
                        os.makedirs(os.path.join(script_args.output_dir,f"layer{key_flag+1}"))
                    np.save(os.path.join(script_args.output_dir,f"layer{key_flag+1}/round{global_round}_server.npy"), server_array)
                    server_Proj=torch.from_numpy(np_server_Proj)
                    server_Proj=server_Proj.to(device='cuda:0')    
                key_flag+=1
                # #取出每个layer的每个server的P或Q
        else:
            key_flag=0
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                if fed_args.is_quantized:
                    delta_w=compressor.compress(delta_w,quantize_level=fed_args.quantize_level)
                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                opt_proxy_dict[key] = fed_args.fedopt_beta2*opt_proxy_dict[key] + (1-fed_args.fedopt_beta2)*torch.square(delta_w)
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)
    else:   # Normal dataset-size-based aggregation 
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
    

    return global_dict, global_auxiliary, save_subrank, save_gradient_norm, pre_client_delta_w, server_proxy_Proj, client_init