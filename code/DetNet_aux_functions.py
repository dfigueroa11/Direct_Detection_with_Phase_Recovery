import torch

import DD_system
import calc_filters




########################## Funtion to generate the training data ###############################
def one_batch_data_generation(block_len, sym_mem, snr_lin, const):
    #################### System definition ##################
    DD_sys = DD_system.DD_system()
    DD_sys.N_sim = 2
    DD_sys.N_os = 2
    symbol_rate = 35e9
    symbol_time = 1/symbol_rate
    DD_sys.Ts = symbol_time/DD_sys.N_sim
    fs = 1/DD_sys.Ts

    ################# Photo diode definition ################
    DD_sys.sigma_sh = 0
    DD_sys.sigma_th = 1/torch.sqrt(snr_lin)
    DD_sys.responsivity = 1
    DD_sys.on_off_noise = 1

    ################# Channel definition ###################
    alpha_dB_km = 0.2
    beta_2_s2_km = -2.168e-23
    fiber_len_km = 0

    ################# filters creation definition ###################
    rc_alpha = 0
    pulse_shape_len = 101
    channel_filt_len = 1
    rx_filt_len = 1

    g_tx_td = torch.tensor(calc_filters.fd_rc_td(rc_alpha, pulse_shape_len, fs, symbol_time), dtype=torch.cfloat)
    DD_sys.g_tx_td = g_tx_td[None, None,:]
    channel_td = torch.tensor(calc_filters.CD_fiber_td(alpha_dB_km, beta_2_s2_km, fiber_len_km, channel_filt_len, fs), dtype=torch.cfloat)
    DD_sys.channel_td = channel_td[None, None,:]
    g_rx_td = torch.tensor(calc_filters.fd_rc_td(0, rx_filt_len, fs, symbol_time/2), dtype=torch.float64)
    DD_sys.g_rx_td = g_rx_td[None, None,:]

    bits = torch.randint(2,((block_len+sym_mem)*const.m,))
    info_symbols = const.map(bits)
    tx_syms = const.diff_encoding(info_symbols, init_phase_idx=0)

    y = DD_sys.simulate_system_td(tx_syms[None,:],sym_mem-1)

    psi_n = DD_sys.get_auxiliary_equiv_channel(sym_mem).flip(0)
    Psi_mat = torch.zeros(2*block_len, 2*block_len+2*sym_mem, dtype=torch.cfloat)
    for i in range(2*block_len):
        Psi_mat[i,i:i+len(psi_n)] = psi_n
    Psi_e_mat = Psi_mat[::2,1::2]
    Psi_o_mat = Psi_mat[1::2,1::2]

    tx_syms_ml = torch.cat((torch.real(tx_syms),torch.imag(tx_syms)),-1)
    Psi_e_mat_ml = torch.cat((torch.cat((torch.real(Psi_e_mat),-torch.imag(Psi_e_mat)),-1),
                              torch.cat((torch.imag(Psi_e_mat),torch.real(Psi_e_mat)),-1)),-2)
    Psi_o_mat_ml = torch.cat((torch.cat((torch.real(Psi_o_mat),-torch.imag(Psi_o_mat)),-1),
                              torch.cat((torch.imag(Psi_o_mat),torch.real(Psi_o_mat)),-1)),-2)

    return y[0,:2*block_len:2], y[0,1:2*block_len:2], Psi_e_mat_ml, Psi_o_mat_ml, tx_syms_ml

def data_generation(block_len, sym_mem, batch_size, snr_dB, snr_dB_var, const, device):    
    y_e = torch.empty((batch_size,block_len), device=device)
    y_o = torch.empty((batch_size,block_len), device=device)
    Psi_e = torch.empty((batch_size,2*block_len, 2*(block_len+sym_mem)), device=device)
    Psi_o = torch.empty((batch_size,2*block_len, 2*(block_len+sym_mem)), device=device)
    tx_syms = torch.empty((batch_size,2*(block_len+sym_mem)), device=device)

    snr_lin = 10.0 ** ((snr_dB+2*snr_dB_var*(torch.rand(batch_size)-0.5))/10.0)
    for i in range(batch_size):
        y_e[i], y_o[i], Psi_e[i], Psi_o[i], tx_syms[i] = one_batch_data_generation(block_len, sym_mem, snr_lin[i], const)

    return y_e, y_o, Psi_e, Psi_o, tx_syms

############################ One Hot functions #################################

def sym_2_oh():
    pass


def oh_2_sym(mapp_re, mapp_im, syms_oh, syms_len, device):
    mapp_re_len = len(mapp_re)
    mapp_im_len = len(mapp_im)
    syms_oh_re = syms_oh[:,:syms_len*mapp_re_len].reshape(-1, syms_len, mapp_re_len)
    syms_oh_im = syms_oh[:,syms_len*mapp_re_len:].reshape(-1, syms_len, mapp_im_len)
    syms = torch.cat((torch.matmul(syms_oh_re, mapp_re),torch.matmul(syms_oh_im, mapp_im)),1)
    syms.to(device)
    return syms





