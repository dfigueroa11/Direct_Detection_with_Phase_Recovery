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
training_steps = 200
batch_size_train = 1_000

model.train()

results = []
ber = []
ser = []
for i in range(training_steps):
    # Generate a batch of training data
    y_e, y_o, Psi_e, Psi_o, tx_syms = aux_func.data_generation(block_len, sym_mem, batch_size_train, snr_dB, snr_dB_var, const, device)
    tx_syms_oh = aux_func.sym_2_oh(const.mapping_re, const.mapping_im, tx_syms, device) 
    # feed data to the network
    x, x_oh = model(y_e, y_o, Psi_e, Psi_o, const.mapping_re, const.mapping_im)
    # compute loss
    loss = torch.sum(aux_func.layer_loss_paper_learning_to_detect(x_oh, tx_syms_oh, device))
    # compute gradients
    loss.backward()
    # Adapt weights
    optimizer.step()
    # reset gradients
    optimizer.zero_grad()

    # Print the current progress of the training (Loss and BER).
    if i%50 == 0 or i == (training_steps-1):       
        results.append(aux_func.layer_loss_paper_learning_to_detect(x_oh, tx_syms_oh, device).detach().cpu().numpy())
        sym_idx_train = const.nearest_neighbor(tx_syms[:,:sym_len]+1j*tx_syms[:,sym_len:]).detach().cpu()
        sym_idx_DetNet = const.nearest_neighbor(x[-1,:,:sym_len]+1j*x[-1,:,sym_len:]).detach().cpu()
        bits_train = const.demap(sym_idx_train)
        bits_DetNet = const.demap(sym_idx_DetNet)
        ber.append(ch_met.get_ER(bits_train.flatten(),bits_DetNet.flatten()))
        ser.append(ch_met.get_ER(sym_idx_train.flatten(),sym_idx_DetNet.flatten()))
        print(f'Train step {i:_}\t\tcurrent loss: {results[-1][-1]}\t\tBER: {ber[-1]}\t\tSER: {ser[-1]}')
        x_aux = x[-1,:,:sym_len]+1j*x[-1,:,sym_len:]
        mean_error_vector = torch.mean(torch.min(torch.abs(x_aux.flatten().unsqueeze(1)-const.mapping),1)[0])
        print(torch.abs(const.mapping))
        print(mean_error_vector)


    del y_e, y_o, Psi_e, Psi_o, tx_syms
    torch.cuda.empty_cache()

x_aux = x_aux.flatten().detach()
plt.figure()
plt.hist(tx_syms_oh.flatten().detach().numpy())
plt.figure()
plt.hist(x_oh[-1].flatten().detach().numpy())
plt.figure()
plt.scatter(torch.real(x_aux),torch.imag(x_aux))
plt.show()

torch.save(model.state_dict(), 'DetNet_test.pt')





