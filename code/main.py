import numpy as np
import DD_system
import matplotlib.pyplot as plt


system = DD_system.DD_system()
system.N_sim = 31
system.beta_2 = -2.168e-23
system.fiber_alpha = 0.2
system.fiber_length = 30
system.symbol_rate = 15e9
system.sigma_sh = 0
system.sigma_th = 0
system.symbol_time = 1/system.symbol_rate
system.Ts = system.symbol_time/system.N_sim
system.Bw = 1/system.Ts
system.pulse_shape = np.exp(-(np.arange(-100,101))**2/60)
system.responsivity = 1
system.on_off = 1


signal = system.pulse_shaping([1])
plt.figure(0)
plt.plot(signal)

signal = system.applay_channel(signal)

signal = system.square_law_detection(signal)
plt.plot(signal/np.max(signal))


plt.show()
