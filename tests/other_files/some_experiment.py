import torch
from Params import configs

device = torch.device(configs.device)

candidate  = torch.tensor([0,6,12,18,24,30]).to(device)
candidate = candidate.unsqueeze(0)
print(candidate)
#print(candidate.size)
print(candidate.shape)
print(candidate.size(0))
# print(candidate.size(1))
print("################################################")
dummy = candidate.unsqueeze(-1)
print(candidate)
print(dummy)
dummy = candidate.unsqueeze(-1).expand(-1, 6, 64)
print(dummy)