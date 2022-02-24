# 用于寻找6x6instance出现bug的问题
# import torch
# a = torch.randn(1,1)
#
# print(a)
# print(a.shape)
# print("###################")
# print(a.squeeze())
# print(a.squeeze().shape)
# print("###################")
# print(a.view(-1))
# print(a.view(-1).shape)
# print("###################")
# # print(a.resize(1,))
# # print(a.resize(1,).shape)
# # print("###################")
# # print(a.resize_as(1,))
# # print(a.resize_as(1,).shape)
# print(a.view(-1).std())   # 关键就是这里啊

#############update中的这一句导致了nan值的出现#############
# rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)     # reward 还作为一个归一化
# rewards只有一个数，然后rewards.std导致了0的dan

import numpy as np
np.random.seed(300)
MTBF = 50
sampling =np.random.exponential(MTBF,10000)
print(sampling.shape)
print(np.mean(sampling))