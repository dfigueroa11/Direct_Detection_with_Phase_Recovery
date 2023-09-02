import torch
from torch.nn.functional import pad
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

import DD_system
import calc_filters
import constellation
import bcjr_upsamp
import channel_metrics as ch_met
import constellation_maker as const_mk

#################### System definition ##################
DD_sys = DD_system.DD_system()
DD_sys.N_sim = 2
DD_sys.N_os = 2
symbol_rate = 35e9
symbol_time = 1/symbol_rate
DD_sys.Ts = symbol_time/DD_sys.N_sim
fs = 1/DD_sys.Ts

################# Photo diode definition ################
DD_sys.sigma_sh = 0
DD_sys.sigma_th = 1
DD_sys.responsivity = 1
DD_sys.on_off_noise = 1

################# Channel definition ###################
alpha_dB_km = 0.2
beta_2_s2_km = -2.168e-23
fiber_len_km = 0

################# filters creation definition ###################
rc_alpha = 0
pulse_shape_len = 101
channel_filt_len = 101
rx_filt_len = 1

g_tx_td = torch.tensor(calc_filters.fd_rc_td(rc_alpha, pulse_shape_len, fs, symbol_time), dtype=torch.cfloat)
DD_sys.g_tx_td = g_tx_td[None, None,:]
channel_td = torch.tensor(calc_filters.CD_fiber_td(alpha_dB_km, beta_2_s2_km, fiber_len_km, channel_filt_len, fs), dtype=torch.cfloat)
DD_sys.channel_td = channel_td[None, None,:]
g_rx_td = torch.tensor(calc_filters.fd_rc_td(0, rx_filt_len, fs, symbol_time/2), dtype=torch.float64)
DD_sys.g_rx_td = g_rx_td[None, None,:]

############# Constellation and differential mapping ################
mapping_DDSQAM = torch.tensor(const_mk.rp_QAM(np.array([1]),np.array([0,1.23095942,np.pi,np.pi+1.23095942])), dtype=torch.cfloat)
diff_mapping_DDSQAM = torch.tensor([[1,0,3,2],[0,1,2,3],[3,2,1,0],[2,3,0,1]])

mapping_QAM = torch.tensor(const_mk.rp_QAM(np.array([1]),np.array([1*np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])), dtype=torch.cfloat)
diff_mapping_QAM = torch.tensor([[2,3,0,1],[3,0,1,2],[0,1,2,3],[1,2,3,0]])

mapping_BPSK = torch.tensor(const_mk.rp_QAM(np.array([1]),np.array([0,np.pi])), dtype=torch.cfloat)
diff_mapping_BPSK = torch.tensor([[1,0],[0,1]])

################# Simulation definition ####################
file_name = 'my_constellation.pkl'
N_symbols = 20_000
mapping_list = [mapping_DDSQAM, mapping_QAM, mapping_BPSK]
diff_mapping_list = [diff_mapping_DDSQAM, diff_mapping_QAM, diff_mapping_BPSK]
SNR_dB_list = [*range(-5,14)]
sym_mem_aux_ch_list = [7]

#################### Simulation #########################

ser = -torch.ones((len(mapping_list),len(sym_mem_aux_ch_list),len(SNR_dB_list)), dtype=torch.float)
for i, (mapping, diff_mapping) in enumerate(zip(mapping_list,diff_mapping_list)):
    print(f'start simulation for constellation = {mapping}')
    for j, sym_mem_aux_ch in enumerate(sym_mem_aux_ch_list):
        print(f'\tstart simulation for memory = {sym_mem_aux_ch}')
        for k, SNR_dB in enumerate(SNR_dB_list):
            print(f'\t\tstart simulation for SNR = {SNR_dB}')
            start = time.time()

            SNR_lin = 10**(SNR_dB/10)
            mapping *= torch.sqrt(SNR_lin/torch.mean(torch.abs(mapping)**2))
            const = constellation.constellation(mapping,'cpu',diff_mapping)

            bits = torch.randint(2,(N_symbols*const.m,))
            info_symbols = const.map(bits)
            ch_symbols_1 = const.diff_encoding(info_symbols, init_phase_idx=0)

            y_1 = DD_sys.simulate_system_td(pad(ch_symbols_1, (sym_mem_aux_ch,0), 'constant', 0)[None,:])

            decoder = bcjr_upsamp.bcjr_upsamp(DD_sys.get_auxiliary_equiv_channel(sym_mem_aux_ch), 0, N_symbols, const, DD_sys.N_os, diff_decoding=True)
            beliefs = decoder.compute_true_apps(y_1, log_out=False, P_s0=None)#(torch.eye(const.M)[0:1,:]-1)*1e10)

            symbols_hat_idx = torch.argmax(beliefs[0,0:], dim=1)
            bits_hat = const.demap(symbols_hat_idx)
            symbols_hat = const.map(bits_hat.int())

            ser[i,j,k] = ch_met.get_ER(info_symbols, symbols_hat)
            
            end = time.time()
            print(f"\t\tsimulation time: {end-start: .3f} s")

            ############################### Save results ########################## 
            # (done always, so in case the simulation stops the data is not lost)
            results = {key:value for key, value in DD_sys.__dict__.items() if not key.startswith('__') and not callable(key)}
            results['rc_alpha'] = rc_alpha
            results['alpha_dB_km'] =  alpha_dB_km
            results['beta_2_s2_km'] = beta_2_s2_km
            results['fiber_len_km'] = fiber_len_km
            results['mapping'] = mapping
            results['diff_mapping'] = diff_mapping
            results['N_symbols'] = N_symbols
            results['SNR_dB_list'] = SNR_dB_list
            results['sym_mem_aux_ch_list'] = sym_mem_aux_ch_list
            results['ser'] = ser

            with open(file_name, 'wb') as f:
                pickle.dump(results, f)
            ###########################################################################

print(ser)
