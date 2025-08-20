import torch
import numpy as np
import math



import time






class QSGDCompressor(object):
    # 定义QSGD压缩器类
    def __init__(self):
        self.name = 'qsgd'  # 定义压缩器的名称为'qsgd'
        self.residuals = {}  # 初始化一个空字典，用于存储各个张量的残差
        self.values = {}  # 初始化一个空字典，用于存储各个张量的压缩后的值
        self.zc = None  # 初始化zc属性为None，可能用于后续的某些操作
        self.current_ratio = 1  # 设置当前压缩比率为1，表示初始状态下没有压缩
        self.shapes = {}  # 初始化一个空字典，用于存储各个张量的形状信息

    def get_qsgd(self, x, s, is_biased=False):
        # 定义QSGD (Quantized Stochastic Gradient Descent) 方法
        norm = x.norm(p=2)  # 计算张量x的2范数（L2范数）
        
        level_float = s * x.abs() / norm  # 计算量化值：将张量x的每个元素绝对值乘以s，然后除以范数
        
        previous_level = torch.floor(level_float)  # 向下取整，得到每个元素的量化下界
        
        # 添加随机量化：生成与x形状相同的随机张量，用于决定是否将量化值向上调整一个级别
        is_next_level = (torch.rand_like(x) < (level_float - previous_level)).float()
        
        new_level = previous_level + is_next_level  # 最终的量化级别：previous_level加上随机调整值

        scale = 1  # 初始化缩放因子
        
        if is_biased:
            d = x.nelement()  # 获取张量x中元素的总数
            scale = 1.0 / (min(d / (s ** 2), math.sqrt(d) / s) + 1.0)  # 计算QSGD的方差界，用于减小量化误差
        
        # 返回量化后的结果
        return scale * torch.sign(x) * norm * new_level / s

    def qsgd_quantize_numpy(self, x, s, is_biased=False):
        """在绝对值系数上对张量x进行d级量化"""
        norm = np.sqrt(np.sum(np.square(x)))  # 计算x的2范数（L2范数）
        
        level_float = s * np.abs(x) / norm  # 计算量化值：将x的每个元素绝对值乘以s，然后除以范数
        
        previous_level = np.floor(level_float)  # 向下取整，得到每个元素的量化下界
        
        # 添加随机量化：生成与x形状相同的随机数组，用于决定是否将量化值向上调整一个级别
        is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
        
        new_level = previous_level + is_next_level  # 最终的量化级别：previous_level加上随机调整值

        scale = 1  # 初始化缩放因子为1
        
        if is_biased:
            d = len(x)  # 获取x的长度
            scale = 1.0 / (np.minimum(d / s ** 2, np.sqrt(d) / s) + 1.0)  # 计算QSGD的方差界，用于减小量化误差
        
        # 返回量化后的结果
        return scale * np.sign(x) * norm * new_level / s

    def compress(self, tensor, name=None, quantize_level=32, is_biased=True):
        # 定义压缩方法，接受张量、名称、量化级别和是否有偏置作为参数
        if quantize_level != 32:
            s = 2 ** quantize_level - 1  # 计算量化参数s，为2的量化级别次方减1
            values = self.get_qsgd(tensor, s, is_biased)  # 使用QSGD方法进行压缩
        else:
            values = tensor  # 如果量化级别为32，则不进行压缩
        return values  # 返回压缩后的值（或未压缩的原始张量）

    def decompress_new(self, tensor):
        # 这个方法用于解压缩操作
        return tensor  # 直接返回输入的tensor，不进行任何处理

    def update_shapes_dict(self, tensor, name):
        # 更新shapes字典，记录不同名称的张量的形状信息
        self.shapes[name] = tensor.shape  # 将传入的tensor的形状存储到self.shapes字典中，以name为键





class TopKCompressor():
    """
    Sparse Communication for Distributed Gradient Descent, Alham Fikri Aji et al., 2017
    """
    def __init__(self):
        self.residuals = {}
        self.sparsities = []
        self.zero_conditions = {}
        self.values = {} 
        self.indexes = {} 
        self.c = 0
        self.t = 0.
        self.name = 'topk'
        self.zc = None
        self.current_ratio = 1
        # ===================================
        self.shapes = {}


    def _process_data_before_selecting(self, name, data):
        pass

    def _process_data_after_residual(self, name, data):
        if name not in self.zero_conditions:
            self.zero_conditions[name] = torch.ones(data.numel(), dtype=torch.float32, device=data.device) 
        zero_condition = self.zero_conditions[name]
        zero_condition.fill_(1.0)
        zero_condition[self.indexes[name]] = 0.0
        self.zc = zero_condition

    def clear(self):
        self.residuals = {}
        self.sparsities = []
        self.zero_conditions = {}
        self.values = {} 
        self.indexes = {} 


    # def compress(self, tensor, name=None, sigma_scale=2.5, ratio=0.05):
    #     start = time.time()
    #     with torch.no_grad():
    #         if name not in self.residuals:
    #             self.residuals[name] = torch.zeros_like(tensor.data)
    #         # top-k solution
    #         numel = tensor.numel()
    #         k = max(int(numel * ratio), 1)
    #         self.current_ratio = ratio
    #         #tensor.data.add_(TopKCompressor.residuals[name].data)
    #         self._process_data_before_selecting(name, tensor.data)

    #         values, indexes = torch.topk(torch.abs(tensor.data), k=k)
    #         values = tensor.data[indexes]

    #         self.residuals[name].data = tensor.data + 0.0 
    #         self.residuals[name].data[indexes] = 0. 
    #         self.values[name] = values
    #         self.indexes[name] = indexes

    #         self._process_data_after_residual(name, tensor.data)

    #         return tensor, indexes, values

    def compress(self, tensor, name=None, sigma_scale=2.5, ratio=0.05):
        start = time.time()
        with torch.no_grad():
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio

            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values = tensor.data[indexes]

            self.values[name] = values
            self.indexes[name] = indexes

            return tensor, indexes, values

    def decompress(self, tensor, original_tensor_size):
        return tensor


    def decompress_new(self, tensor, indexes, name=None, shape=None):
        '''
            Just decompress, without unflatten.
            Remember to do unflatter after decompress
        '''
        if shape is None:
            decompress_tensor = torch.zeros(
                self.shapes[name], dtype=tensor.dtype, device=tensor.device).view(-1)
            decompress_tensor[indexes] = tensor
            # decompress_tensor = torch.zeros(self.shapes[name]).view(-1)
            # decompress_tensor[indexes] = tensor.type(decompress_tensor.dtype)
            return decompress_tensor
        else:
            decompress_tensor = torch.zeros(
                self.shapes[name], dtype=tensor.dtype, device=tensor.device).view(-1)
            decompress_tensor[indexes] = tensor
            # decompress_tensor = torch.zeros(self.shapes[name]).view(-1)
            # decompress_tensor[indexes] = tensor.type(decompress_tensor.dtype)
            return decompress_tensor

    def flatten(self, tensor, name=None):
        ''' 
            flatten a tensor 
        '''
        self.shapes[name] = tensor.shape
        return tensor.view(-1)

    def unflatten(self, tensor, name=None, shape=None):
        ''' 
            unflatten a tensor 
        '''
        if shape is None:
            return tensor.view(self.shapes[name])
        else:
            return tensor.view(shape)

    def update_shapes_dict(self, tensor, name):
        self.shapes[name] = tensor.shape

    def get_residuals(self, name, like_tensor):
        if name not in self.residuals:
            self.residuals[name] = torch.zeros_like(like_tensor.data)
        return self.residuals[name]

    def add_residuals(self, included_indexes, name):
        with torch.no_grad():
            residuals = self.residuals[name]
            if type(included_indexes) is np.ndarray:
                indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            else:
                indexes_t = included_indexes
            values = self.values[name]
            values.data[indexes_t] = 0.0
            residuals.data[self.indexes[name]] += values.data
            #selected_indexes = TopKCompressor.indexes[name][indexes_t]
            #residuals.data[selected_indexes] = 0.0 
            #logger.info('residuals after: %f', torch.norm(TopKCompressor.residuals[name].data))





