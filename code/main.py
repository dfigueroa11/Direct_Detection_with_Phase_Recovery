import numpy as np
import DD_system
import matplotlib.pyplot as plt


system = DD_system.DD_system()
system.N_sim = 4
system.beta_2 = -2.168e-23
system.fiber_alpha = 0.2
system.fiber_length = 30
system.symbol_rate = 35e9
system.sigma_sh = 0
system.sigma_th = 0
system.symbol_time = 1/system.symbol_rate
system.Ts = system.symbol_time/system.N_sim
system.Bw = 1/system.Ts
system.pulse_shape = np.sinc(np.arange(-10,10,0.25))
print(np.arange(-2.5,2.75,0.25))
system.responsivity = 1
system.on_off = 1


signal = system.pulse_shaping([1]*1)
plt.figure(0)
plt.stem(signal)

signal = system.applay_channel(signal)

signal = system.square_law_detection(signal)
plt.stem(signal/np.max(signal),markerfmt='+')


plt.show()
