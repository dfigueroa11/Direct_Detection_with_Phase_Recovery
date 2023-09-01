import torch
from torch.nn.functional import pad
import numpy as np
import matplotlib.pyplot as plt

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
rc_alpha = 0

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
pulse_shape_len = 101
channel_filt_len = 101
rx_filt_len = 1

g_tx_td = torch.tensor(calc_filters.fd_rc_td(rc_alpha, pulse_shape_len, fs, symbol_time), dtype=torch.cfloat)
DD_sys.g_tx_td = g_tx_td[None, None,:]
channel_td = torch.tensor(calc_filters.CD_fiber_td(alpha_dB_km, beta_2_s2_km, fiber_len_km, channel_filt_len, fs), dtype=torch.cfloat)
DD_sys.channel_td = channel_td[None, None,:]
g_rx_td = torch.tensor(calc_filters.fd_rc_td(0, rx_filt_len, fs, symbol_time/2), dtype=torch.float64)
DD_sys.g_rx_td = g_rx_td[None, None,:]

################# Simulation definition ####################
sym_mem_aux_ch = 1
N_symbols = 20_000

###################### Constellation #########################


SNR_dB = 12


mapping = torch.tensor(const_mk.rp_QAM(np.array([1]),np.array([0,1.23095942,np.pi,np.pi+1.23095942])), dtype=torch.cfloat)
diff_mapping = torch.tensor([[1,0,3,2],[0,1,2,3],[3,2,1,0],[2,3,0,1]])
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

symbols_hat_idx_0 = torch.nonzero(symbols_hat_idx==0)
symbols_hat_idx_1 = torch.nonzero(symbols_hat_idx==1)
symbols_hat_idx_2 = torch.nonzero(symbols_hat_idx==2)
symbols_hat_idx_3 = torch.nonzero(symbols_hat_idx==3)




print(f"BER = {ch_met.get_ER(bits, bits_hat): .3f}")
print(f"SER = {ch_met.get_ER(info_symbols, symbols_hat): .3f}")

print(torch.angle(const.mapping))
print(f'error idx0 -> idx2: {torch.count_nonzero(info_symbols[symbols_hat_idx_0]==const.mapping[2])}')
print(f'error idx0 -> idx1: {torch.count_nonzero(info_symbols[symbols_hat_idx_0]==const.mapping[1])}')
print(f'error idx0 -> idx3: {torch.count_nonzero(info_symbols[symbols_hat_idx_0]==const.mapping[3])}')
print(f'error idx1 -> idx0: {torch.count_nonzero(info_symbols[symbols_hat_idx_1]==const.mapping[0])}')
print(f'error idx1 -> idx2: {torch.count_nonzero(info_symbols[symbols_hat_idx_1]==const.mapping[2])}')
print(f'error idx1 -> idx3: {torch.count_nonzero(info_symbols[symbols_hat_idx_1]==const.mapping[3])}')
print(f'error idx2 -> idx0: {torch.count_nonzero(info_symbols[symbols_hat_idx_2]==const.mapping[0])}')
print(f'error idx2 -> idx1: {torch.count_nonzero(info_symbols[symbols_hat_idx_2]==const.mapping[1])}')
print(f'error idx2 -> idx3: {torch.count_nonzero(info_symbols[symbols_hat_idx_2]==const.mapping[3])}')
print(f'error idx3 -> idx0: {torch.count_nonzero(info_symbols[symbols_hat_idx_3]==const.mapping[0])}')
print(f'error idx3 -> idx1: {torch.count_nonzero(info_symbols[symbols_hat_idx_3]==const.mapping[1])}')
print(f'error idx3 -> idx2: {torch.count_nonzero(info_symbols[symbols_hat_idx_3]==const.mapping[2])}')

# # for i in range(const.M):
# plt.figure(0)
#     # plt.stem(y_1[0,:], markerfmt='o',label='1')
#     # plt.scatter(range(len(beliefs[0,:,i])),beliefs[0,:,i])
# # # plt.grid()
# # plt.legend()

y_idx = torch.arange(1,len(y_1[0]),2)
symbols_idx_0 = torch.nonzero(info_symbols==const.mapping[0])
symbols_idx_1 = torch.nonzero(info_symbols==const.mapping[1])
symbols_idx_2 = torch.nonzero(info_symbols==const.mapping[2])
symbols_idx_3 = torch.nonzero(info_symbols==const.mapping[3])
lim_min = float(torch.min(y_1))
lim_max = float(torch.max(y_1))

plt.figure(0)
plt.hist(y_1[0,y_idx[symbols_idx_0[:,0]]],bins=40,range=(lim_min,lim_max), alpha=0.3, label='0')
plt.hist(y_1[0,y_idx[symbols_idx_1[:,0]]],bins=40,range=(lim_min,lim_max), alpha=0.3, label='1.23')
plt.hist(y_1[0,y_idx[symbols_idx_2[:,0]]],bins=40,range=(lim_min,lim_max), alpha=0.3, label='pi')
plt.hist(y_1[0,y_idx[symbols_idx_3[:,0]]],bins=40,range=(lim_min,lim_max), alpha=0.3, label='-1.91')

plt.legend()
plt.show()
