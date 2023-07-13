import torch
import DD_system
import matplotlib.pyplot as plt


system = DD_system.DD_system()
system.N_sim = 31
system.beta_2 = -2.168e-23
system.fiber_length = 30
system.symbol_rate = 35e9
system.sigma_sh = 0.
system.sigma_th = 0.
system.symbol_time = 1/system.symbol_rate
system.Ts = system.symbol_time/system.N_sim
system.Bw = 1/system.Ts
system.pulse_shape = torch.exp(-(torch.arange(-100,101))**2/200)
system.responsivity = 1
system.on_off = 1


signal = system.applay_channel(torch.tensor([1]*4))
plt.figure(0)

signal = system.square_law_detection(signal)

plt.plot(signal)

plt.show()
