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
one_hot_len_mag = len(const.mag_list)
one_hot_len_phase = len(const.phase_list)

model = DetNet_architecture.DetNet(layers, block_len, sym_mem, one_hot_len_mag, one_hot_len_phase, v_len, z_len, device)
model.to(device)

#################### Adam Optimizer ############################
optimizer = optim.Adam(model.parameters(), eps=1e-07)

###################### Training ################################
# hyperparameters
training_steps = 1000
batch_size_train = 200

model.train()

results = []
for i in range(training_steps):
    # Generate a batch of training data
    y_e, y_o, Psi_e, Psi_o, tx_syms = aux_func.data_generation(block_len, sym_mem, batch_size_train, snr_dB, snr_dB_var, const, device)
    tx_syms_re = tx_syms[:,:sym_len]
    tx_syms_im = tx_syms[:,sym_len:]
    tx_mag = torch.sqrt(torch.square(tx_syms_re)+torch.square(tx_syms_im))
    tx_phase_diff = torch.diff(torch.atan2(tx_syms_im,tx_syms_re), prepend=torch.zeros(batch_size_train,1, device=device), dim=-1)
    # feed data to the network
    x_mag, x_phase = model(y_e, y_o, Psi_e, Psi_o, const.mag_list, const.phase_list)
    
    # compute loss
    x_phase_diff = torch.diff(x_phase, prepend=torch.zeros(layers,batch_size_train,1, device=device), dim=-1)
    loss = torch.sum(aux_func.per_layer_loss_distance_square(x_mag, tx_mag, device)) + \
           torch.sum(aux_func.per_layer_loss_distance_square(torch.cos(x_phase_diff), torch.cos(tx_phase_diff), device))
    
    # compute gradients
    loss.backward()
    # Adapt weights
    optimizer.step()
    # reset gradients
    optimizer.zero_grad()

    # Print and save the current progress of the training
    if i%(training_steps//20) == 0 or i == (training_steps-1):       
        results.append(aux_func.per_layer_loss_distance_square(x_mag, tx_mag, device).detach().cpu().numpy())
        results.append(aux_func.per_layer_loss_distance_square(torch.cos(x_phase_diff), torch.cos(tx_phase_diff), device).detach().cpu().numpy())
        print(f'Train step {i:_}\n\tcurrent mag loss:\t{results[-2][-1]}\n\tcurrent phase loss:\t{results[-1][-1]}')
        x = (x_mag*torch.exp(1j*x_phase))
        mean_error_vector_x = torch.mean(torch.min(torch.abs(x.flatten().unsqueeze(1)-const.mapping),1)[0])
        print(f"\tEVM of x: {mean_error_vector_x}")
        
        x = x.flatten().detach().cpu()
        plt.figure()
        plt.scatter(torch.real(x),torch.imag(x), label='x')
        plt.legend()
        plt.xlim((-2.5,2.5))
        plt.ylim((-2.5,2.5))
        plt.grid()
        plt.savefig(f'../../results/scatter_x_hat_trainstep{i}.pdf', dpi=20)
        plt.figure()
        plt.hist(x_mag.flatten().detach().cpu())
        plt.savefig(f'../../results/hist_x_mag_trainstep{i}.pdf', dpi=20)        
        plt.figure()
        plt.hist(x_phase.flatten().detach().cpu())
        plt.savefig(f'../../results/hist_x_phase_trainstep{i}.pdf', dpi=20)
        plt.close('all')
        torch.save(model.state_dict(), '../../results/DetNet_test.pt')
        del x, mean_error_vector_x


    del y_e, y_o, Psi_e, Psi_o, tx_syms
    torch.cuda.empty_cache()

torch.save(model.state_dict(), '../../results/DetNet_test.pt')





