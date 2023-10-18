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
snr_dB = 12
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
model.load_state_dict(torch.load('../../results/magPhase_DetNet_v2/DetNet_test.pt', map_location=torch.device(device)))
model.to(device)
model.eval()

###################### Testing ################################
batch_size = 5_000


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
x_phase_diff = aux_func.diff_decoding(x_phase, angle, device)

ber.append(aux_func.get_ber(x_mag[-1,:,:-1], x_phase_diff[-1,:,1:], tx_mag[:,:-1], tx_phase[:,1:], const))
ser.append(aux_func.get_ser(x_mag[-1,:,:-1], x_phase_diff[-1,:,1:], tx_mag[:,:-1], tx_phase[:,1:], const))
print(f"\tBER:\t\t\t{ber[-1]:}")
print(f"\tSER:\t\t\t{ser[-1]}")

x_diff = (x_mag[-1,:,:-1]*torch.exp(1j*x_phase_diff[-1,:,1:]))
x_diff = x_diff.flatten().detach().cpu()


fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
ax.scatter(torch.real(x_diff),torch.imag(x_diff), marker='o', s=15, c='b', label='MagPhase DetNet', alpha=0.5)
ax.plot(np.cos(np.linspace(0, 2 * np.pi, 100)), np.sin(np.linspace(0, 2 * np.pi, 100)), 'k--', alpha=0.5)
ax.scatter(np.cos([0, angle, np.pi, np.pi + angle]), np.sin([0, angle, np.pi, np.pi + angle]), marker='o', s=70, c='red', label='Constellation Points')
line_angles = [np.pi / 2, np.pi - angle / 2, np.pi + angle / 2, 3 * np.pi / 2, -angle / 2, angle / 2]
for a in line_angles:
    ax.plot([0, 4 * np.cos(a)], [0, 4 * np.sin(a)], 'g:', linewidth=2, label="Decision Boundaries")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:3], labels[:3], loc=1)
ax.set_xlabel('Re')
ax.set_ylabel('Im')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.grid(True)
plt.show()





print("ok")





