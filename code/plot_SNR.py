import pickle
import torch
import matplotlib.pyplot as plt

DetNet_results = torch.load(f'../../results/magPhase_DetNet_testing.pt',
                                map_location=torch.device('cpu'))


results = []
file_name = 'SER_N14_BPSK.pkl'
with open(file_name, 'rb') as file:
    results.append(pickle.load(file))
file_name = 'SER_N14_BPSK_ideal.pkl'
with open(file_name, 'rb') as file:
    results.append(pickle.load(file))
file_name = 'SER_N7_DDSQAM_QAM.pkl'
with open(file_name, 'rb') as file:
    results.append(pickle.load(file))
file_name = 'SER_N7_DDSQAM_QAM_ideal.pkl'
with open(file_name, 'rb') as file:
    results.append(pickle.load(file))

SNR_dB = results[0]['SNR_dB_list']
SNR_dB_ideal = results[-1]['SNR_dB_list']

plt.figure(0)
plt.semilogy(SNR_dB,results[0]['ser'][0,0,:], label=f'BPSK, N={14:d}, M=50')
plt.semilogy(SNR_dB_ideal,results[1]['ser'][0,0,:], label=f'BPSK, N={14:d}, M=14')
plt.semilogy(SNR_dB,results[2]['ser'][0,0,:], label=f'DD-SQAM, N={7:d}, M=50')
plt.semilogy(SNR_dB_ideal,results[3]['ser'][0,0,:], label=f'DD-SQAM, N={7:d}, M=7')
plt.semilogy(SNR_dB,results[2]['ser'][1,0,:], label=f'QAM, N={7:d}, M=50')
plt.semilogy(SNR_dB_ideal,results[3]['ser'][1,0,:], label=f'QAM, N={7:d}, M=7')
plt.ylim((0.001,1))
plt.xlim((-5,13))
plt.grid(which='both')
plt.legend()
plt.title(r'SER for B2B configuration and raised cosine $\alpha = 0$ $M$ = 101')
plt.ylabel('SER')
plt.xlabel('SNR [dB]')
