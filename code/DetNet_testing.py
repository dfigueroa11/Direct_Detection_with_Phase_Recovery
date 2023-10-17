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
import DetNet_architecture


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)


# System config
sym_mem = 1
ch_mem = 2*sym_mem+1
block_len = 4
sym_len = block_len+sym_mem
snr_dB = 20
snr_dB_var = 0

############# Constellation and differential mapping ################
angle = np.arccos(1/3)
mapping = torch.tensor(const_mk.rp_QAM(np.array([1]),np.array([0,angle,np.pi,np.pi+angle])), dtype=torch.cfloat)
diff_mapping = torch.tensor([[1,0,3,2],[0,1,2,3],[3,2,1,0],[2,3,0,1]])
mapping *= torch.sqrt(1/torch.mean(torch.abs(mapping)**2))
const = constellation.constellation(mapping, device ,diff_mapping)

############################ DetNet declaration #####################
layers = 30#*sym_len
v_len = 2*sym_len
z_len = 4*sym_len
one_hot_len_mag = len(const.mag_list)
one_hot_len_phase = len(const.phase_list)

model = DetNet_architecture.DetNet(layers, block_len, sym_mem, one_hot_len_mag, one_hot_len_phase, v_len, z_len, device)
model.load_state_dict(torch.load('DetNet_test.pt', map_location=torch.device(device)))
model.to(device)
model.eval()

###################### Testing ################################
batch_size = 500


ber = []
ser = []
# Generate a batch of training data
y_e, y_o, Psi_e, Psi_o, tx_syms = aux_func.data_generation(block_len, sym_mem, batch_size, snr_dB, snr_dB_var, const, device)
tx_syms_re = tx_syms[:,:sym_len]
tx_syms_im = tx_syms[:,sym_len:]
tx_mag = torch.sqrt(torch.square(tx_syms_re)+torch.square(tx_syms_im))
tx_phase = torch.atan2(tx_syms_im,tx_syms_re)
# feed data to the network
s = time.time()
x_mag, x_phase = model(y_e, y_o, Psi_e, Psi_o, const.mag_list, const.phase_list)
e = time.time()
print(f"time: {e-s}")
# compute loss
x_phase_diff = torch.diff(x_phase, prepend=torch.zeros(layers,batch_size,1, device=device), dim=-1)

ber.append(aux_func.get_ber(x_mag[-1,:,:-1], x_phase_diff[-1,:,1:], tx_mag[:,:-1], tx_phase[:,1:], const))
ser.append(aux_func.get_ser(x_mag[-1,:,:-1], x_phase_diff[-1,:,1:], tx_mag[:,:-1], tx_phase[:,1:], const))
print(f"\tBER:\t\t\t{ber[-1]}")
print(f"\tSER:\t\t\t{ser[-1]}")

x_diff = (x_mag[-1,:,:-1]*torch.exp(1j*x_phase_diff[-1,:,1:]))
x_diff = x_diff.flatten().detach().cpu()
tx = (tx_mag[:,:-1]*torch.exp(1j*tx_phase[:,1:]))
tx = tx.flatten().detach().cpu()

plt.figure(figsize=(6,6))
plt.scatter([1,-1,np.cos(angle),-np.cos(angle),np.cos(angle),-np.cos(angle)],[0,0,np.sin(angle),np.sin(angle),-np.sin(angle),-np.sin(angle)])
plt.plot([0,4*np.cos(angle/2)],[0,4*np.sin(angle/2)], 'g--')
plt.plot([0,0],[0,4], 'g--')
plt.plot([0,-4*np.cos(angle/2)],[0,4*np.sin(angle/2)], 'g--')
plt.plot([0,-4*np.cos(angle/2)],[0,-4*np.sin(angle/2)], 'g--')
plt.plot([0,4*np.cos(angle/2)],[0,-4*np.sin(angle/2)], 'g--')
plt.plot([0,0],[0,-4], 'g--')
plt.scatter(torch.real(x_diff),torch.imag(x_diff), label='rx')
plt.scatter(torch.real(tx),torch.imag(tx), label='tx')
plt.legend()
plt.xlim((-1.5,1.5))
plt.ylim((-1.5,1.5))
plt.grid()
plt.show()





print("ok")





