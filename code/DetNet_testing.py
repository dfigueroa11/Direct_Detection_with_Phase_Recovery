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

model_checkpoint = torch.load('../../results_sym_mem1/magphase_DetNet_test.pt', map_location=torch.device(device))
# System config
sym_mem = model_checkpoint['sym_mem']
block_len = model_checkpoint['block_len']
ch_mem = 2*sym_mem+1
sym_len = block_len+sym_mem

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
N_symbols = 10_000
N_frames = 50
batch_size = N_symbols//N_frames
used_symbols = 1

snr_dB_list = [20,]
snr_dB_var = 0

ser = []
time_decoding = []
# generate the Psi_e Psi_o matrix for decoding, SNR values are not important, only dimensions are.


for snr_dB in snr_dB_list:
    ser_aux = 0
    time_deco = 0
    for i in range(N_frames):
        # generate data
        y_e, y_o, Psi_e, Psi_o, tx_mag, tx_phase, state_mag, state_phase = aux_func.data_generation(block_len, sym_mem, batch_size,
                                                                                        snr_dB, snr_dB_var, const, device)
        
        start = time.time()
        #decode
        rx_mag, rx_phase = magphase_DetNet(y_e, y_o, Psi_e, Psi_o, state_mag, state_phase, layers, return_all=False)
        rx_syms = rx_mag[:,:used_symbols]*torch.exp(1j*rx_phase[:,:used_symbols])
        rx_syms_idx = const.nearest_neighbor(rx_syms)
        end = time.time()
        time_deco += end - start

        # compere with tx symbols to get the SER
        tx_syms = tx_mag[:,:used_symbols]*torch.exp(1j*tx_phase[:,:used_symbols])
        tx_syms_idx = const.nearest_neighbor(tx_syms)
        ser_aux += ch_met.get_ER(tx_syms_idx.flatten(),rx_syms_idx.flatten())/N_frames
    
    ser.append(ser_aux) 
    time_decoding.append(time_deco)

print(ser)
print(time_decoding)
