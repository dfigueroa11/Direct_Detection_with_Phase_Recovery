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
N_symbols = 5

if sim_in_time_domain:
    pulse_shape_len = 11
    channel_filt_len = 11
    rx_filt_len = 11

    DD_sys.g_tx_td = torch.tensor([[calc_filters.fd_rc_td(rc_alpha, pulse_shape_len, fs, symbol_time)]], dtype=torch.complex128)
    DD_sys.channel_td = torch.tensor([[calc_filters.CD_fiber_td(alpha_dB_km, beta_2_s2_km, fiber_len_km, channel_filt_len, fs)]], dtype=torch.complex128)
    DD_sys.g_rx_td = torch.tensor([[calc_filters.fd_rc_td(0, rx_filt_len, fs, symbol_time/2)]], dtype=torch.float64)
if sim_in_freq_domain:
    min_zero_padd = 300

    DD_sys.len_fft = int(2**np.ceil(np.log2(N_symbols*DD_sys.N_sim+min_zero_padd)))
    DD_sys.G_tx_fd = calc_filters.fd_rc_fd(rc_alpha, DD_sys.len_fft, fs, symbol_time)
    DD_sys.channel_fd = calc_filters.CD_fiber_fd(alpha_dB_km, beta_2_s2_km, fiber_len_km, DD_sys.len_fft, fs)
    DD_sys.G_rx_fd = calc_filters.fd_rc_fd(0, DD_sys.len_fft, fs, symbol_time/2) 


symbols = torch.zeros(1,30, dtype=torch.complex128)
symbols[0,15] = 1
sym = torch.tensor([[3, 0, 1, 0, 0, 1, 3, 3, 2, 1, 3, 0, 3, 1, 0, 1, 1, 3, 0, 0, 0, 0, 2, 2, 0, 1, 0, 2, 3, 2]], dtype=torch.complex128)
x = DD_sys.simulate_system_fd(sym)
print(x.size())
# test 1
# sign_1 = np.random.randint(2, size=N_symbols)*2-1
# sign_2 = np.random.randint(2, size=N_symbols)*2-1

# d_angles = np.random.random(size=N_symbols)*2*np.pi
# symbols_1 = np.exp(1j*np.cumsum(d_angles*sign_1))
# symbols_2 = np.exp(1j*np.cumsum(d_angles*sign_2))

# d_symbols_1 = symbols_1[1:]/symbols_1[:-1]
# d_symbols_2 = symbols_2[1:]/symbols_2[:-1]

# test 2
# constellation = torch.tensor([1, -1, np.sqrt(2)/2+np.sqrt(2)/2*1j, -np.sqrt(2)/2-np.sqrt(2)/2*1j])

# index = torch.randint(4, size=(1,N_symbols))
# symbols_1 = constellation[index]
# symbols_2 = constellation[(index+2)%4]
# symbols_3 = -constellation[index]
# symbols_4 = -constellation[(index+2)%4]

# xd = np.random.randint(2, size=N_symbols)*2-1
# x_1 = DD_sys.simulate_system_td(symbols_1)
# x_2 = DD_sys.simulate_system_td(symbols_2)
# x_3 = DD_sys.simulate_system_td(symbols_3)
# x_4 = DD_sys.simulate_system_td(symbols_4)

plt.figure(0)
plt.stem(x[0,:], markerfmt='o', label='fd')
# plt.stem(x_2, markerfmt='*', label='td')
# plt.stem(x_3, markerfmt='o', label='fd')
# plt.stem(x_4, markerfmt='*', label='td')
# plt.legend()

# plt.figure(1)
# plt.stem(np.angle(symbols_1)/np.pi, markerfmt='o')
# plt.stem(np.angle(symbols_2)/np.pi, markerfmt='*')
# plt.figure(2)
# plt.stem(DD_sys.g_tx_td, markerfmt='o')
# plt.plot(np.imag(np.fft.fftshift(DD_sys.G_tx_fd*calc_filters.CD_fiber_fd(alpha_dB_km=0.2, beta_2_s2_km=-2.168e-23, fiber_len_km=30, len=DD_sys.len_fft, fs=fs))))


# plt.stem(signal/np.max(signal),markerfmt='+')


plt.show()