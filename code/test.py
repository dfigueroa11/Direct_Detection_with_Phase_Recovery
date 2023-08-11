import torch
import numpy as np
from itertools import product
import bcjr
import constellation

mapping = torch.tensor([1,-1], dtype=torch.cfloat)
const = constellation.constellation(mapping,'cpu')
taps = torch.tensor([2,3,5,7], dtype=torch.cfloat)
EsN0_dB = 0
block_len = 6

bits = torch.tensor([1,0,1,0,1,1,0])#torch.randint(2,(block_len,))
block_len = len(bits)
y = np.convolve(const.map(bits),taps)
y = torch.tensor(y)
y = y[None,:]
decoder = bcjr.bcjr(taps, EsN0_dB, block_len, const)
beliefs = decoder.compute_true_apps(y, log_out=False)

print(bits)
print(beliefs)
