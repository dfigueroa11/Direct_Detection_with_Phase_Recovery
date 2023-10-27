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
import MagPhaseDetNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)

resume_training = False
if resume_training:
    checkpoint = torch.load('../../results3/results/magphase_DetNet_test.pt', map_location=torch.device(device))


# System config
if resume_training:
    sym_mem = checkpoint['sym_mem']
    block_len = checkpoint['block_len']
else:
    sym_mem = 1
    block_len = 4

sym_len = block_len+sym_mem


############# Constellation and differential mapping ################
angle = np.arccos(1/3)
mapping = torch.tensor(const_mk.rp_QAM(np.array([1]),np.array([0,angle,np.pi,np.pi+angle])), dtype=torch.cfloat)
diff_mapping = torch.tensor([[1,0,3,2],[0,1,2,3],[3,2,1,0],[2,3,0,1]])
mapping *= torch.sqrt(1/torch.mean(torch.abs(mapping)**2))
const = constellation.constellation(mapping, device ,diff_mapping)
############################ DetNet declaration #####################
if resume_training:
    layers = checkpoint['layers']
    v_len = checkpoint['v_len']
    z_len = checkpoint['z_len']
    mag_list = checkpoint['mag_list']
    phase_list = checkpoint['phase_list']
else:
    layers = max(3*sym_len,30)
    v_len = 2*sym_len
    z_len = 4*sym_len
    mag_list = const.mag_list
    phase_list = const.phase_list

magphase_DetNet = MagPhaseDetNet.MagPhaseDetNet(layers, block_len, sym_mem, mag_list, phase_list, v_len, z_len, device)
magphase_DetNet.angle = angle

if resume_training:
    magphase_DetNet.mag_model.load_state_dict(checkpoint['mag_state_dict'])
    magphase_DetNet.phase_model.load_state_dict(checkpoint['phase_state_dict'])
    
#################### Adam Optimizer ############################
mag_optimizer = optim.Adam(magphase_DetNet.mag_model.parameters(), eps=1e-07)
phase_optimizer = optim.Adam(magphase_DetNet.phase_model.parameters(), eps=1e-07)

if resume_training:
    mag_optimizer.load_state_dict(checkpoint['mag_optimizer'])
    phase_optimizer.load_state_dict(checkpoint['phase_optimizer'])

###################### Training ################################
# hyperparameters
batches_per_epoch = 500
batch_size_per_epoch = [100,200,400,600,1000,2000]
snr_dB_list = [17,17,17,15,13,12]
snr_dB_var_list = [3,]*len(batch_size_per_epoch)
images_per_epoch = 3
cnt = checkpoint['cnt'] if resume_training else 0


if resume_training:
    window_phase = checkpoint['window_phase']
else:
    window_phase = torch.arange(sym_mem, -1, -1, dtype=torch.float, device=device)*2 + 1
    if block_len > sym_mem + 1:
        window_phase = pad(window_phase, (block_len-len(window_phase),0), 'constant', window_phase[0])

magphase_DetNet.train()    

results = []
ber = []
ser = []

# for i in range(training_steps):
for batch_size, snr_dB, snr_dB_var in zip(batch_size_per_epoch, snr_dB_list, snr_dB_var_list):
    for i in range(batches_per_epoch):
        # Generate a batch of training data
        y_e, y_o, Psi_e, Psi_o, tx_mag, tx_phase, state_mag, state_phase = aux_func.data_generation(block_len, sym_mem, batch_size,
                                                                                                    snr_dB, snr_dB_var, const, device)
        # feed data to the network
        rx_mag, rx_phase = magphase_DetNet(y_e, y_o, Psi_e, Psi_o, state_mag, state_phase, layers)
        
        
        mag_loss = torch.sum(aux_func.per_layer_loss_distance_square(rx_mag, tx_mag, device))
        phase_loss = torch.sum(aux_func.per_layer_loss_distance_square(torch.abs(rx_phase), torch.abs(tx_phase), device, window_phase))

        # compute gradients
        mag_loss.backward(retain_graph=True)
        phase_loss.backward()
        # Adapt weights
        mag_optimizer.step()
        phase_optimizer.step()
        # reset gradients
        mag_optimizer.zero_grad()
        phase_optimizer.zero_grad()
        # Print and save the current progress of the training
        if (i+1)%(batches_per_epoch//images_per_epoch) == 0:  
            results.append(aux_func.per_layer_loss_distance_square(rx_mag, tx_mag, device).detach().cpu().numpy())
            results.append(aux_func.per_layer_loss_distance_square(torch.abs(rx_phase), torch.abs(tx_phase), device, window_phase).detach().cpu().numpy())
            print(f'Batch size {batch_size:_}, Train step {i:_}, cnt {cnt}\n\tcurrent mag loss:\t{results[-2][-1]}\n\tcurrent phase loss:\t{results[-1][-1]}')
            for j in range(block_len):
                ber.append(aux_func.get_ber(rx_mag[-1,:,j:j+1], rx_phase[-1,:,j:j+1], tx_mag[:,j:j+1], tx_phase[:,j:j+1], const))
                ser.append(aux_func.get_ser(rx_mag[-1,:,j:j+1], rx_phase[-1,:,j:j+1], tx_mag[:,j:j+1], tx_phase[:,j:j+1], const))
                print(f"\tBER for symbol {j+1}:\t{ber[-1]}")
                print(f"\tSER for symbol {j+1}:\t{ser[-1]}")
                rx = (rx_mag[-1,:,j:j+1]*torch.exp(1j*rx_phase[-1,:,j:j+1]))
                
                rx = rx.flatten().detach().cpu()

                fig = plt.figure(figsize=(6,6))
                ax = fig.add_subplot(1,1,1)
                ax.scatter(torch.real(rx),torch.imag(rx), marker='o', s=15, c='b', label='MagPhase DetNet', alpha=0.5)
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
                plt.savefig(f'../../results/scatter_rx_hat{j}_{cnt}.pdf')
                plt.close('all')

            cnt +=1     
            checkpoint = {'mag_state_dict': magphase_DetNet.mag_model.state_dict(),
                          'mag_optimizer': mag_optimizer.state_dict(),
                          'phase_state_dict': magphase_DetNet.phase_model.state_dict(),
                          'phase_optimizer': phase_optimizer.state_dict(),
                          'results': results,
                          'cnt': cnt,
                          'layers': layers,
                          'block_len': block_len,
                          'sym_mem': sym_mem,
                          'mag_list': const.mag_list,
                          'phase_list': const.phase_list,
                          'v_len': v_len,
                          'z_len': z_len,
                          'window_phase': window_phase}
            torch.save(checkpoint, '../../results/magphase_DetNet_test.pt')
            del rx

        del y_e, y_o, Psi_e, Psi_o, tx_mag, tx_phase
        torch.cuda.empty_cache()

checkpoint = {'mag_state_dict': magphase_DetNet.mag_model.state_dict(),
              'mag_optimizer': mag_optimizer.state_dict(),
              'phase_state_dict': magphase_DetNet.phase_model.state_dict(),
              'phase_optimizer': phase_optimizer.state_dict(),
              'results': results,
              'cnt': cnt,
              'layers': layers,
              'block_len': block_len,
              'sym_mem': sym_mem,
              'mag_list': mag_list,
              'phase_list': phase_list,
              'v_len': v_len,
              'z_len': z_len,
              'window_phase': window_phase}
torch.save(checkpoint, '../../results/magphase_DetNet_test.pt')
                        