import numpy as np
import matplotlib.pyplot as plt

import DD_system
import calc_filters


DD_sys = DD_system.DD_system()
DD_sys.N_sim = 4
DD_sys.N_os = 2
DD_sys.sigma_sh = 0
DD_sys.sigma_th = 0
DD_sys.responsivity = 1
DD_sys.on_off_noise = 1


symbol_rate = 35e9
symbol_time = 1/symbol_rate
Ts = symbol_time/DD_sys.N_sim
fs = 1/Ts

N_symbols = 100
min_zero_padd = 100

DD_sys.len_fft = int(2**np.ceil(np.log2(N_symbols*DD_sys.N_sim+min_zero_padd)))
DD_sys.G_tx_fd = calc_filters.fd_rc_fd(alpha=0.2, fs=fs, sym_time=symbol_time, len=DD_sys.len_fft)
DD_sys.channel_fd = calc_filters.CD_fiber_fd(alpha_dB_km=0.2, beta_2_s2_km=-2.168e-23, fiber_len_km=30, len=DD_sys.len_fft, fs=fs)
DD_sys.G_rx_fd = calc_filters.fd_rc_fd(alpha=0, fs=fs, sym_time=symbol_time/2, len=DD_sys.len_fft)

symbols = np.zeros(N_symbols)
symbols[50] = 1
xd = DD_sys.simulate_system_fd(symbols)


# signal = system.pulse_shaping([1]*1)
plt.figure(0)
plt.stem(xd)

# signal = system.applay_channel(signal)

# signal = system.square_law_detection(signal)
# plt.stem(signal/np.max(signal),markerfmt='+')


plt.show()
