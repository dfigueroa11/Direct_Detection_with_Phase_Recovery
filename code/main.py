import torch
import numpy as np
import matplotlib.pyplot as plt

import DD_system
import calc_filters

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
DD_sys.sigma_th = 0
DD_sys.responsivity = 1
DD_sys.on_off_noise = 1

################# Channel definition ###################
alpha_dB_km = 0.2
beta_2_s2_km = -2.168e-23
fiber_len_km = 0

################# Simulation definition ####################
sim_in_time_domain = True
sim_in_freq_domain = True
N_symbols = 50

if sim_in_time_domain:
    pulse_shape_len = 3
    channel_filt_len = 11
    rx_filt_len = 11

    g_tx_td = torch.tensor(calc_filters.fd_rc_td(rc_alpha, pulse_shape_len, fs, symbol_time), dtype=torch.complex128)
    DD_sys.g_tx_td = g_tx_td[None, None,:]
    channel_td = torch.tensor(calc_filters.CD_fiber_td(alpha_dB_km, beta_2_s2_km, fiber_len_km, channel_filt_len, fs), dtype=torch.complex128)
    DD_sys.channel_td = channel_td[None, None,:]
    g_rx_td = torch.tensor(calc_filters.fd_rc_td(0, rx_filt_len, fs, symbol_time/2), dtype=torch.float64)
    DD_sys.g_rx_td = g_rx_td[None, None,:]
if sim_in_freq_domain:
    min_zero_padd = 300

    DD_sys.len_fft = int(2**np.ceil(np.log2(N_symbols*DD_sys.N_sim+min_zero_padd)))
    DD_sys.G_tx_fd = calc_filters.fd_rc_fd(rc_alpha, DD_sys.len_fft, fs, symbol_time)
    DD_sys.channel_fd = calc_filters.CD_fiber_fd(alpha_dB_km, beta_2_s2_km, fiber_len_km, DD_sys.len_fft, fs)
    DD_sys.G_rx_fd = calc_filters.fd_rc_fd(0, DD_sys.len_fft, fs, symbol_time/2) 


# test 1
sign_1 = torch.randint(2, size=(1,N_symbols))*2-1
sign_2 = torch.randint(2, size=(1,N_symbols))*2-1

d_angles = torch.rand(size=(N_symbols,))*2*np.pi
symbols_1 = torch.exp(1j*torch.cumsum(d_angles*sign_1, dim=1)).type(torch.complex128)
symbols_2 = torch.exp(1j*torch.cumsum(d_angles*sign_2, dim=1)).type(torch.complex128)
d_symbols_1 = symbols_1[1:]/symbols_1[:-1]
d_symbols_2 = symbols_2[1:]/symbols_2[:-1]


x_1 = DD_sys.simulate_system_td(symbols_1)
x_2 = DD_sys.simulate_system_td(symbols_2)

plt.figure(0)
plt.stem(torch.real(x_1[0,:]), markerfmt='o', label='fd')
plt.stem(torch.real(x_2[0,:]), markerfmt='*', label='td')
# plt.legend()



plt.show()