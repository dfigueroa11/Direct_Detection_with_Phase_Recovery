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
snr_dB = 8
snr_dB_var = 4

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
model.to(device)

#################### Adam Optimizer ############################
optimizer = optim.Adam(model.parameters(), eps=1e-07)

###################### Training ################################
# hyperparameters
# training_steps = 20_000
# batch_size_train = 200
batches_per_epoch = 1_500
batch_size_per_epoch = [100,200,300,600,800,1_000,2_000,5_000]#np.linspace(10,10_000,num=num_epochs).astype(int)
cnt = 0

mag_loss_weight = 1e-2
phase_loss_weight = 1 - mag_loss_weight


    
model.train()

results = []
ber = []
ser = []
# for i in range(training_steps):
for batch_size in batch_size_per_epoch:
    for i in range(batches_per_epoch):
        # Generate a batch of training data
        y_e, y_o, Psi_e, Psi_o, tx_syms = aux_func.data_generation(block_len, sym_mem, batch_size, snr_dB, snr_dB_var, const, device)
        tx_syms_re = tx_syms[:,:sym_len]
        tx_syms_im = tx_syms[:,sym_len:]
        tx_mag = torch.sqrt(torch.square(tx_syms_re)+torch.square(tx_syms_im))
        tx_phase = torch.atan2(tx_syms_im,tx_syms_re)
        # feed data to the network
        x_mag, x_phase = model(y_e, y_o, Psi_e, Psi_o, const.mag_list, const.phase_list)
        
        # compute loss
        x_phase_diff = aux_func.diff_decoding(x_phase, angle, device)
        loss = mag_loss_weight*torch.sum(aux_func.per_layer_loss_distance_square(x_mag[:,:,:-1], tx_mag[:,:-1], device)) + \
            phase_loss_weight*torch.sum(aux_func.per_layer_loss_distance_square(torch.abs(x_phase_diff[:,:,1:]), torch.abs(tx_phase[:,1:]), device))
        
        # compute gradients
        loss.backward()
        # Adapt weights
        optimizer.step()
        # reset gradients
        optimizer.zero_grad()

        # Print and save the current progress of the training
        if (i+1)%(batches_per_epoch//3) == 0:  
            results.append(aux_func.per_layer_loss_distance_square(x_mag[:,:,:-1], tx_mag[:,:-1], device).detach().cpu().numpy())
            results.append(aux_func.per_layer_loss_distance_square(torch.abs(x_phase_diff[:,:,1:]), torch.abs(tx_phase[:,1:]), device).detach().cpu().numpy())
            ber.append(aux_func.get_ber(x_mag[-1,:,:-1], x_phase_diff[-1,:,1:], tx_mag[:,:-1], tx_phase[:,1:], const))
            ser.append(aux_func.get_ser(x_mag[-1,:,:-1], x_phase_diff[-1,:,1:], tx_mag[:,:-1], tx_phase[:,1:], const))
            print(f'Batch size {batch_size:_}, Train step {i:_}\n\tcurrent mag loss:\t{results[-2][-1]}\n\tcurrent phase loss:\t{results[-1][-1]}')
            print(f"\tBER:\t\t\t{ber[-1]}")
            print(f"\tSER:\t\t\t{ser[-1]}")
            x_diff = (x_mag[-1,:,:-1]*torch.exp(1j*x_phase_diff[-1,:,1:]))
            
            x_diff = x_diff.flatten().detach().cpu()
            plt.figure()
            plt.scatter(torch.real(x_diff),torch.imag(x_diff), label='x')
            plt.legend()
            plt.xlim((-2.5,2.5))
            plt.ylim((-2.5,2.5))
            plt.grid()
            plt.savefig(f'../../results/scatter_x_diff_hat_{cnt}.pdf', dpi=20)
            plt.close('all')
            checkpoint = {'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'results': results}
            torch.save(checkpoint, '../../results/DetNet_test.pt')
            cnt +=1     
            del x_diff

        del y_e, y_o, Psi_e, Psi_o, tx_syms
        torch.cuda.empty_cache()

checkpoint = {'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
torch.save(checkpoint, '../../results/DetNet_test.pt')
            




