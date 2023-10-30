import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time 

import DD_system
import calc_filters
import constellation
import bcjr_upsamp
import channel_metrics as ch_met
import constellation_maker as const_mk
import DetNet_aux_functions as aux_func
import MagPhaseDetNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)

model_checkpoint = torch.load('../../results_w2_bl9/magphase_DetNet_test.pt', map_location=torch.device(device))

# System config
sym_mem = model_checkpoint['sym_mem']
block_len = model_checkpoint['block_len']
ch_mem = 2*sym_mem+1
sym_len = block_len+sym_mem

snr_dB_list = [1,]
snr_dB_var = 0

############# Constellation and differential mapping ################
angle = np.arccos(1/3)
mapping = torch.tensor(const_mk.rp_QAM(np.array([1]),np.array([0,angle,np.pi,np.pi+angle])), dtype=torch.cfloat)
diff_mapping = torch.tensor([[1,0,3,2],[0,1,2,3],[3,2,1,0],[2,3,0,1]])
mapping *= torch.sqrt(1/torch.mean(torch.abs(mapping)**2))
const = constellation.constellation(mapping, device ,diff_mapping)

############################ DetNet declaration #####################
layers = model_checkpoint['layers']
v_len = model_checkpoint['v_len']
z_len = model_checkpoint['z_len']
mag_list = model_checkpoint['mag_list']
phase_list = model_checkpoint['phase_list']


magphase_DetNet = MagPhaseDetNet.MagPhaseDetNet(layers, block_len, sym_mem, mag_list, phase_list, v_len, z_len, device)
magphase_DetNet.angle = angle
magphase_DetNet.mag_model.load_state_dict(model_checkpoint['mag_state_dict'])
magphase_DetNet.phase_model.load_state_dict(model_checkpoint['phase_state_dict'])
    
magphase_DetNet.eval()

###################### Testing ################################
N_symbols = 300
ser = []

# generate the Psi_e Psi_o matrix for decoding, SNR values are not important, only dimensions are.
_, _, Psi_e, Psi_o, _, _, _, _ = aux_func.data_generation(block_len, sym_mem, 1, 0, 0, const, device)


for snr_dB in snr_dB_list:
    y_e, y_o, _, _, tx_mag, tx_phase, state_mag, state_phase = aux_func.data_generation(N_symbols+block_len-1, sym_mem, 1,
                                                                                        snr_dB, snr_dB_var, const, device)
    tx_mag = tx_mag[:,:-sym_mem]
    tx_phase = tx_phase[:,:-sym_mem]
    rx_mag = torch.empty_like(tx_mag)
    rx_phase = torch.empty_like(tx_phase)
    print('xd')
    s = time.time()
    for i in range(N_symbols):
        mag, phase = magphase_DetNet(y_e[:,i:i+block_len], y_o[:,i:i+block_len], Psi_e, Psi_o, state_mag, state_phase, layers, return_all=False)
        #update state
        rx_mag[:,i] = mag [:,0]
        rx_mag[:,i] = mag [:,0]
    e = time.time()
    print(e-s)

