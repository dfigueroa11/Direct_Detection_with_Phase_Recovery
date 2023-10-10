import torch
import torch.nn as nn
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
import DetNet_architecture


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)


# System config
sym_mem = 2
ch_mem = 2*sym_mem+1
block_len = 4
sym_len = block_len+sym_mem
snr_dB = 10
snr_dB_var = 2

############# Constellation and differential mapping ################
angle = np.arccos(1/3)
mapping = torch.tensor(const_mk.rp_QAM(np.array([1]),np.array([0,angle,np.pi,np.pi+angle])-angle/2), dtype=torch.cfloat)
diff_mapping = torch.tensor([[1,0,3,2],[0,1,2,3],[3,2,1,0],[2,3,0,1]])
mapping *= torch.sqrt(1/torch.mean(torch.abs(mapping)**2))
const = constellation.constellation(mapping, device ,diff_mapping)

############################ DetNet declaration #####################
layers = 30#*sym_len
v_len = 2*sym_len
z_len = 4*sym_len
one_hot_len = len(const.mapping_re) + len(const.mapping_im)

model = DetNet_architecture.DetNet(layers, block_len, sym_mem, one_hot_len, v_len, z_len, device)
model.load_state_dict(torch.load('../../results/DetNet_test.pt', map_location=torch.device(device)))
model.to(device)
model.eval()

###################### Testing ################################
tasting_steps = 1
batch_size_train = 5


results = []
for i in range(tasting_steps):
    # Generate a batch of training data
    y_e, y_o, Psi_e, Psi_o, tx_syms = aux_func.data_generation(block_len, sym_mem, batch_size_train, snr_dB, snr_dB_var, const, device)
    # feed data to the network
    x, x_tilde = model(y_e, y_o, Psi_e, Psi_o, const.mapping_re, const.mapping_im)
    
    u_tx = tx_syms
    u_hat = x[-1]
    u_tx_comp = u_tx[:,:sym_len]+1j*u_tx[:,sym_len:]
    u_hat_comp = u_hat[:,:sym_len]+1j*u_hat[:,sym_len:]

    print(aux_func.sql_detection(u_tx, Psi_e, Psi_o, device))
    print(aux_func.sql_detection(u_hat, Psi_e, Psi_o, device))
    print(torch.angle(u_tx_comp))
    print(torch.angle(u_hat_comp))
    
    del y_e, y_o, Psi_e, Psi_o, tx_syms
    torch.cuda.empty_cache()






