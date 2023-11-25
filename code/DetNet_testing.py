import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time 
import pickle

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

N_symbols = 100_000
N_frames = 250
batch_size = N_symbols//N_frames
used_symbols = 1

snr_dB_list = [*range(0,21)]
sym_mem_file_list = [1,3,5]

############# Constellation and differential mapping ################
angle = np.arccos(1/3)
mapping = torch.tensor(const_mk.rp_QAM(np.array([1]),np.array([0,angle,np.pi,np.pi+angle])), dtype=torch.cfloat)
diff_mapping = torch.tensor([[1,0,3,2],[0,1,2,3],[3,2,1,0],[2,3,0,1]])
mapping *= torch.sqrt(1/torch.mean(torch.abs(mapping)**2))
const = constellation.constellation(mapping, device ,diff_mapping)


ser = -torch.ones((len(sym_mem_file_list),len(snr_dB_list)), device=device)
sim_time = -torch.ones((len(sym_mem_file_list),len(snr_dB_list)), device=device)

for sym_mem_idx, sym_mem_file in enumerate(sym_mem_file_list):
    print(f'simulation for symbol memory {sym_mem_file}')
    for snr_idx, snr_dB in enumerate(snr_dB_list):
        print(f'\tSNR {snr_dB}')
        checkpoint = torch.load(f'../../final_sym_mem_{sym_mem_file}/magphase_Det_Net_sym_mem_{sym_mem_file}_snr_{snr_dB}.pt',
                                        map_location=torch.device(device))

        # System config
        sym_mem = checkpoint['sym_mem']
        block_len = checkpoint['block_len']
        sym_len = block_len+sym_mem

        ############################ DetNet declaration #####################
        layers = checkpoint['layers']
        v_len = checkpoint['v_len']
        z_len = checkpoint['z_len']
        mag_list = checkpoint['mag_list']
        phase_list = checkpoint['phase_list']

        magphase_DetNet = MagPhaseDetNet.MagPhaseDetNet(layers, block_len, sym_mem, mag_list, phase_list, v_len, z_len, device)
        magphase_DetNet.angle = angle

        magphase_DetNet.mag_model.load_state_dict(checkpoint['mag_state_dict'])
        magphase_DetNet.phase_model.load_state_dict(checkpoint['phase_state_dict'])
            
        magphase_DetNet.eval()
        with torch.no_grad():
                
        ###################### Testing ################################
            ser_aux = 0
            sim_time_aux = 0
            for i in range(N_frames):
                # generate data
                if i%(N_frames//10) == 0:
                    print(f'\t\t Frame {i}')
                y_e, y_o, Psi_e, Psi_o, tx_mag, tx_phase, state_mag, state_phase = aux_func.data_generation(block_len, sym_mem, batch_size,
                                                                                                snr_dB, 0, const, device)
                
                start = time.time()
                #decode
                rx_mag, rx_phase = magphase_DetNet(y_e, y_o, Psi_e, Psi_o, state_mag, state_phase, layers, return_all=False)
                rx_syms = rx_mag[:,:used_symbols]*torch.exp(1j*rx_phase[:,:used_symbols])
                rx_syms_idx = const.nearest_neighbor(rx_syms)
                end = time.time()

                sim_time_aux += end-start
                # compere with tx symbols to get the SER
                tx_syms = tx_mag[:,:used_symbols]*torch.exp(1j*tx_phase[:,:used_symbols])
                tx_syms_idx = const.nearest_neighbor(tx_syms)
                ser_aux += ch_met.get_ER(tx_syms_idx.flatten(),rx_syms_idx.flatten())/N_frames
                del y_e, y_o, Psi_e, Psi_o, tx_mag, tx_phase, state_mag, state_phase, rx_mag, rx_phase
                torch.torch.cuda.empty_cache()
            
            ser[sym_mem_idx, snr_idx] = ser_aux
            sim_time[sym_mem_idx, snr_idx] = sim_time_aux
            
            results = {'ser': ser,
                       'sim_time': sim_time}
            torch.save(results, f'../../results/magPhase_DetNet_testing.pt')

print(ser)
print(sim_time)

results = {'ser': ser,
           'sim_time': sim_time}
torch.save(results, f'../../results/magPhase_DetNet_testing.pt')
