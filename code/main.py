import torch
import numpy as np
import matplotlib.pyplot as plt

import DD_system
import calc_filters
import constellation
import bcjr_upsamp

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
pulse_shape_len = 3
channel_filt_len = 1
rx_filt_len = 1

g_tx_td = torch.tensor(calc_filters.fd_rc_td(rc_alpha, pulse_shape_len, fs, symbol_time), dtype=torch.cfloat)
DD_sys.g_tx_td = g_tx_td[None, None,:]
channel_td = torch.tensor(calc_filters.CD_fiber_td(alpha_dB_km, beta_2_s2_km, fiber_len_km, channel_filt_len, fs), dtype=torch.cfloat)
DD_sys.channel_td = channel_td[None, None,:]
g_rx_td = torch.tensor(calc_filters.fd_rc_td(0, rx_filt_len, fs, symbol_time/2), dtype=torch.float64)
DD_sys.g_rx_td = g_rx_td[None, None,:]

################# Simulation definition ####################
N_symbols = 20_000

###################### Constellation #########################


SNR_dB = 5


mapping = torch.tensor([1,-1], dtype=torch.cfloat)
SNR_lin = 10**(SNR_dB/10)
mapping *= torch.sqrt(SNR_lin/torch.mean(torch.abs(mapping)**2))
const = constellation.constellation(mapping,'cpu')

bits = torch.randint(2,(N_symbols*const.m,))

symbols = torch.concat((mapping[0:1],const.map(bits)))

y_1 = DD_sys.simulate_system_td(symbols[None,:])

decoder = bcjr_upsamp.bcjr_upsamp(DD_sys.g_tx_td[0,0,:], 0, N_symbols+1, const, DD_sys.N_os)
beliefs = decoder.compute_true_apps(y_1, log_out=False, P_s0=(torch.eye(const.M)[0:1,:]-1)*1e10)

print(DD_sys.g_tx_td)
print(beliefs)
print(torch.sum(bits != torch.argmax(beliefs[0,1:], dim=1))/N_symbols)

plt.figure()
# plt.stem(y_1[0,::2])
plt.plot(beliefs[0,:,0])

plt.show()
