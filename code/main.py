import numpy as np
import matplotlib.pyplot as plt

import DD_system
import calc_filters


DD_sys = DD_system.DD_system()
DD_sys.N_sim = 2
DD_sys.N_os = 2
DD_sys.sigma_sh = 0
DD_sys.sigma_th = 0
DD_sys.responsivity = 1
DD_sys.on_off_noise = 1


symbol_rate = 35e9
symbol_time = 1/symbol_rate
DD_sys.Ts = symbol_time/DD_sys.N_sim
fs = 1/DD_sys.Ts

N_symbols = 24
min_zero_padd = 100

DD_sys.len_fft = int(2**np.ceil(np.log2(N_symbols*DD_sys.N_sim+min_zero_padd)))
DD_sys.G_tx_fd = calc_filters.fd_rc_fd(alpha=0.2, fs=fs, sym_time=symbol_time, len=DD_sys.len_fft)
DD_sys.channel_fd = calc_filters.CD_fiber_fd(alpha_dB_km=0.2, beta_2_s2_km=-2.168e-23, fiber_len_km=0, len=DD_sys.len_fft, fs=fs)
DD_sys.G_rx_fd = calc_filters.fd_rc_fd(alpha=0, fs=fs, sym_time=symbol_time/2, len=DD_sys.len_fft)

DD_sys.g_tx_td = calc_filters.fd_rc_td(alpha=0, fs=fs, sym_time=symbol_time, len=5)
DD_sys.channel_td = calc_filters.CD_fiber_td(alpha_dB_km=0.2, beta_2_s2_km=-2.168e-23, fiber_len_km=0, len=101, fs=fs)
DD_sys.g_rx_td = calc_filters.fd_rc_td(alpha=0, fs=fs, sym_time=symbol_time/2, len=1)

sign_1 = np.random.randint(2, size=N_symbols)*2-1
sign_2 = np.random.randint(2, size=N_symbols)*2-1

d_angles = np.random.randint(4, size=N_symbols)*2*np.pi/4
symbols_1 = np.exp(1j*np.cumsum(d_angles*sign_1))
symbols_2 = np.exp(1j*np.cumsum(d_angles*sign_2))

d_symbols_1 = symbols_1[1:]/symbols_1[:-1]
d_symbols_2 = symbols_2[1:]/symbols_2[:-1]

x_1 = DD_sys.simulate_system_td(d_symbols_1)
x_2 = DD_sys.simulate_system_td(d_symbols_2)
# signal = system.pulse_shaping([1]*1)
plt.figure(0)
plt.stem(x_1, markerfmt='o')
plt.stem(x_2, markerfmt='*')

plt.figure(1)
plt.stem(np.angle(d_symbols_1)/np.pi, markerfmt='o')
plt.stem(np.angle(d_symbols_2)/np.pi, markerfmt='*')
# plt.figure(2)
# plt.stem(DD_sys.g_tx_td, markerfmt='o')
# plt.plot(np.imag(np.fft.fftshift(DD_sys.G_tx_fd*calc_filters.CD_fiber_fd(alpha_dB_km=0.2, beta_2_s2_km=-2.168e-23, fiber_len_km=30, len=DD_sys.len_fft, fs=fs))))

# signal = system.applay_channel(signal)

# signal = system.square_law_detection(signal)
# plt.stem(signal/np.max(signal),markerfmt='+')


plt.show()
