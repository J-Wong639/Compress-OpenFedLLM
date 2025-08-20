import random

def split_dataset(fed_args, script_args, dataset):
    dataset = dataset.shuffle(seed=script_args.seed)        # Shuffle the dataset
    local_datasets = []
    if fed_args.split_strategy == "iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    
    return local_datasets

def get_dataset_this_round(dataset, round, fed_args, script_args):
    # 计算每轮训练需要的样本数量 = 批次大小 * 梯度累积步数 * 最大训练步数
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    
    # 确保采样数量不超过数据集大小
    num2sample = min(num2sample, len(dataset))
    
    # 使用轮次作为随机种子,确保每轮采样结果可复现
    random.seed(round)
    
    # 从数据集中随机采样指定数量的样本索引
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    
    # 根据随机索引选择对应的数据样本
    dataset_this_round = dataset.select(random_idx)

    # 返回本轮训练使用的数据集
    return dataset_this_round