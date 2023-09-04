import pickle
import torch
import matplotlib.pyplot as plt

results = []
N = 2
for i in range(1,N+1):
    file_name = f'SER_N{i:d}_DDSQAM_QAM.pkl'
    with open(file_name, 'rb') as file:
        results.append(pickle.load(file))

SNR_dB = results[0]['SNR_dB_list']

plt.figure(0)
for i in range(1,N+1):
    plt.semilogy(SNR_dB,results[i-1]['ser'][0,0,:], label=f'DD_SQAM, n={i:d}')
for i in range(1,N+1):
    plt.semilogy(SNR_dB,results[i-1]['ser'][1,0,:], label=f'QAM, n={i:d}')

plt.ylim((0.001,1))
plt.grid(which='both')
plt.legend()
plt.show()