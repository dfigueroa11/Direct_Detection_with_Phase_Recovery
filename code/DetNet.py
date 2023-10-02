import torch
import torch.nn as nn
from torch.nn.functional import pad
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import DD_system
import calc_filters
import constellation
import bcjr_upsamp
import channel_metrics as ch_met
import constellation_maker as const_mk
import data_generation_DetNet as data_gen


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)


# System config
sym_mem = 8
ch_mem = 2*sym_mem+1
block_len = 30
snr_dB = 10
snr_dB_var = 2

############# Constellation and differential mapping ################
mapping = torch.tensor(const_mk.rp_QAM(np.array([1]),np.array([0,1.23095942,np.pi,np.pi+1.23095942])), dtype=torch.cfloat)
diff_mapping = torch.tensor([[1,0,3,2],[0,1,2,3],[3,2,1,0],[2,3,0,1]])
mapping *= torch.sqrt(1/torch.mean(torch.abs(mapping)**2))
const = constellation.constellation(mapping, device ,diff_mapping)

batch_size = 1
y_e, y_o, Psi_e, Psi_o, tx_syms = data_gen.data_generation(block_len, sym_mem, batch_size, snr_dB, snr_dB_var, const, device)




