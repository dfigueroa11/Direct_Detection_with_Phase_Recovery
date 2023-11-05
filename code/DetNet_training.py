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
    checkpoint = torch.load('../../results_w2_bl9/magphase_DetNet_test.pt', map_location=torch.device(device))


# System config
if resume_training:
    sym_mem = checkpoint['sym_mem']
    block_len = checkpoint['block_len']
else:
    sym_mem = 5
    block_len = sym_mem+1

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
batches_per_epoch = 300
batch_size_per_epoch = [100, 400, 1_000, 2_000, 5_000, 10_000]
snr_dB_steps = [*range(19,20)]
checkpoint_per_epoch = 5
cnt = checkpoint['cnt'] if resume_training else 0


if resume_training:
    window_phase = checkpoint['window_phase']
    results = checkpoint['results']
else:
    window_phase = torch.arange(sym_mem, -1, -1, dtype=torch.float, device=device)*2 + 1
    if block_len > sym_mem + 1:
        window_phase = pad(window_phase, (block_len-len(window_phase),0), 'constant', window_phase[0])
    results = []


magphase_DetNet.train()    

for snr_dB in snr_dB_steps:
    cnt = 0
    results = []
    print(f'train model for SNR {snr_dB} dB')
    for batch_size in batch_size_per_epoch:
        for i in range(batches_per_epoch):
            # Generate a batch of training data
            y_e, y_o, Psi_e, Psi_o, tx_mag, tx_phase, state_mag, state_phase = aux_func.data_generation(block_len, sym_mem, batch_size,
                                                                                                        snr_dB, 0, const, device)
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
            if (i+1)%(batches_per_epoch//checkpoint_per_epoch) == 0:  
                results.append(aux_func.per_layer_loss_distance_square(rx_mag, tx_mag, device).detach().cpu().numpy())
                results.append(aux_func.per_layer_loss_distance_square(torch.abs(rx_phase), torch.abs(tx_phase), device, window_phase).detach().cpu().numpy())
                print(f'\tBatch size {batch_size:_}, Train step {i:_}, cnt {cnt}\n\tcurrent mag loss:\t{results[-2][-1]}\n\tcurrent phase loss:\t{results[-1][-1]}')
                for j in range(block_len):
                    ber = aux_func.get_ber(rx_mag[-1,:,j:j+1], rx_phase[-1,:,j:j+1], tx_mag[:,j:j+1], tx_phase[:,j:j+1], const)
                    ser = aux_func.get_ser(rx_mag[-1,:,j:j+1], rx_phase[-1,:,j:j+1], tx_mag[:,j:j+1], tx_phase[:,j:j+1], const)
                    print(f"\t\tBER for symbol {j+1}:\t{ber}")
                    print(f"\t\tSER for symbol {j+1}:\t{ser}")
                    
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
                            'window_phase': window_phase,
                            'snr_dB': snr_dB}
                torch.save(checkpoint, f'../../results/magphase_Det_Net_sym_mem_{sym_mem}_snr_{snr_dB}.pt')
                
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
                'window_phase': window_phase,
                'snr_dB': snr_dB}
    torch.save(checkpoint, f'../../results/magphase_Det_Net_sym_mem_{sym_mem}_snr_{snr_dB}.pt')
                        