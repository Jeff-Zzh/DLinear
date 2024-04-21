from .. import DLinear

import torch
class Config:
    def __init__(self, seq_len, pred_len, individual, enc_in):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in
config = Config(seq_len=336, pred_len=96, individual=False, enc_in=8)

# 定义模型结构
model = DLinear.Model(config)

# 加载权重参数
path_linux = '/home/zzh/zzh/Dlinear02/DLinear/checkpoints/Exchange_336_96_DLinear_custom_ftM_sl336_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth'
path_windows = '/checkpoints/Exchange_336_96_DLinear_custom_ftM_sl336_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth'
# 加载状态字典，对应torch.save()
checkpoint = torch.load(path_windows)

# 将状态字典（权重参数）加载到模型中
model.load_state_dict(checkpoint) # checkpoint是一个OrderedDict，包含了模型的状态字典，即模型的各个层的参数值。

# 将模型设置为评估模式（不使用 dropout 和 batch normalization）
model.eval()


