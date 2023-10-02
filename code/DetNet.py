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
import DetNet_aux_functions as aux_func


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)


# System config
sym_mem = 3
ch_mem = 2*sym_mem+1
block_len = 20
snr_dB = 10
snr_dB_var = 2

############# Constellation and differential mapping ################
mapping = torch.tensor(const_mk.rp_QAM(np.array([1]),np.array([0,1.23095942,np.pi,np.pi+1.23095942])), dtype=torch.cfloat)
diff_mapping = torch.tensor([[1,0,3,2],[0,1,2,3],[3,2,1,0],[2,3,0,1]])
mapping *= torch.sqrt(1/torch.mean(torch.abs(mapping)**2))
const = constellation.constellation(mapping, device ,diff_mapping)

################### training hyperparameters #######################
batch_size = 2




y_e, y_o, Psi_e, Psi_o, tx_syms = aux_func.data_generation(block_len, sym_mem, batch_size, snr_dB, snr_dB_var, const, device)
tx_syms_oh = aux_func.sym_2_oh(const.mapping_re, const.mapping_im, tx_syms, device)
tx_syms_hat = aux_func.oh_2_sym(const.mapping_re, const.mapping_im, tx_syms_oh, block_len+sym_mem, device)
