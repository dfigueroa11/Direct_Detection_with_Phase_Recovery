import torch
import numpy as np
from itertools import product
import bcjr_upsamp
import constellation

mapping = torch.tensor([1,-1], dtype=torch.cfloat)
const = constellation.constellation(mapping,'cpu')
taps = torch.tensor([.07,0.1,.5,1,.5,.1,.07], dtype=torch.cfloat)
EsN0_dB = 0

bits = torch.tensor([1,0,1,0,1,1,1,0,1,0,0,1,1,1,1,0])#torch.randint(2,(block_len,))
block_len = len(bits)
y = np.convolve(torch.kron(const.map(bits),torch.eye(2)[-1]),taps)
y = torch.tensor(y)
y = y[None,:]
decoder = bcjr_upsamp.bcjr_upsamp(taps, EsN0_dB, block_len, const, 2)
beliefs = decoder.compute_true_apps(torch.abs(y)**2, log_out=False)

print(bits)
print(torch.argmax(beliefs[0], dim=1))
print(beliefs)
