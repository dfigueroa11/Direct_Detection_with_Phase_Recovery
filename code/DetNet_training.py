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
import DetNet_architecture


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)


# System config
sym_mem = 1
ch_mem = 2*sym_mem+1
block_len = 4
sym_len = block_len+sym_mem
snr_dB = 10
snr_dB_var = 2

############# Constellation and differential mapping ################
mapping = torch.tensor(const_mk.rp_QAM(np.array([1]),np.array([0,1.23095942,np.pi,np.pi+1.23095942])), dtype=torch.cfloat)
diff_mapping = torch.tensor([[1,0,3,2],[0,1,2,3],[3,2,1,0],[2,3,0,1]])
mapping *= torch.sqrt(1/torch.mean(torch.abs(mapping)**2))
const = constellation.constellation(mapping, device ,diff_mapping)

############################ DetNet declaration #####################
layers = 30#*sym_len
v_len = 2*sym_len
z_len = 4*sym_len
one_hot_len = len(const.mapping_re) + len(const.mapping_im)

model = DetNet_architecture.DetNet(layers, block_len, sym_mem, one_hot_len, v_len, z_len, device)
model.to(device)

#################### Adam Optimizer ############################
optimizer = optim.Adam(model.parameters(), eps=1e-07)

###################### Training ################################
# hyperparameters
training_steps = 400
batch_size_train = 200

model.train()

results = []
ber = []
ser = []
for i in range(training_steps):
    # Generate a batch of training data
    y_e, y_o, Psi_e, Psi_o, tx_syms = aux_func.data_generation(block_len, sym_mem, batch_size_train, snr_dB, snr_dB_var, const, device)
    tx_syms_re = tx_syms[:,:sym_len]
    tx_syms_im = tx_syms[:,sym_len:]
    tx_mag = torch.sqrt(torch.square(tx_syms_re)+torch.square(tx_syms_im))
    tx_phase_diff = torch.diff(torch.atan2(tx_syms_im,tx_syms_re), prepend=torch.zeros(batch_size_train,1, device=device), dim=-1)
    # feed data to the network
    x_mag, x_phase = model(y_e, y_o, Psi_e, Psi_o, const.mapping_re, const.mapping_im)
    
    # compute loss
    x_phase_diff = torch.diff(x_phase, prepend=torch.zero(layers,batch_size_train,1, device=device), dim=-1)
    loss = torch.sum(aux_func.per_layer_loss_distance_square(x_mag, tx_mag, device)) + \
           torch.sum(aux_func.per_layer_loss_distance_square(torch.cos(x_phase_diff), torch.cos(tx_phase_diff), device))
    
    # compute gradients
    loss.backward()
    # Adapt weights
    optimizer.step()
    # reset gradients
    optimizer.zero_grad()

    # Print and save the current progress of the training
    # if i%(training_steps//20) == 0 or i == (training_steps-1):       
    #     results.append(aux_func.per_layer_loss_distance_square(u, ux_syms_tilde, device).detach().cpu().numpy())
    #     print(f'Train step {i:_}\tcurrent loss: {results[-1][-1]}')
    #     x_tilde_aux = x_tilde[-1,:,:sym_len]+1j*x_tilde[-1,:,sym_len:]
    #     x_aux = x[-1,:,:sym_len]+1j*x[-1,:,sym_len:]
    #     mean_error_vector_x = torch.mean(torch.min(torch.abs(x_aux.flatten().unsqueeze(1)-const.mapping),1)[0])
    #     mean_error_vector_x_t = torch.mean(torch.min(torch.abs(x_tilde_aux.flatten().unsqueeze(1)-const.mapping),1)[0])
    #     print(f"EVM of x_t: {mean_error_vector_x_t}, \t\tEVM of x: {mean_error_vector_x}")
        
    #     x_aux = x_aux.flatten().detach().cpu()
    #     x_tilde_aux = x_tilde_aux.flatten().detach().cpu()
    #     plt.figure()
    #     plt.scatter(torch.real(x_tilde_aux),torch.imag(x_tilde_aux), label='x_tilde')
    #     plt.scatter(torch.real(x_aux),torch.imag(x_aux), label='x')
    #     plt.legend()
    #     plt.xlim((-2.5,2.5))
    #     plt.ylim((-2.5,2.5))
    #     plt.grid()
    #     plt.savefig(f'../../results/scatter_x_hat_trainstep{i}.pdf', dpi=20)
    #     plt.close('all')
    #     torch.save(model.state_dict(), '../../results/DetNet_test.pt')
    #     del x_tilde_aux, x_aux, mean_error_vector_x, mean_error_vector_x_t


    del y_e, y_o, Psi_e, Psi_o, tx_syms
    torch.cuda.empty_cache()

torch.save(model.state_dict(), '../../results/DetNet_test.pt')





