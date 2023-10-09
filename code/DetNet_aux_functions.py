import torch

import DD_system
import calc_filters

########################## Funtion to generate the training data ###############################
def one_batch_data_generation(block_len, sym_mem, snr_lin, const, device):
    #################### System definition ##################
    DD_sys = DD_system.DD_system(device)
    DD_sys.N_sim = 2
    DD_sys.N_os = 2
    symbol_rate = 35e9
    symbol_time = 1/symbol_rate
    DD_sys.Ts = symbol_time/DD_sys.N_sim
    fs = 1/DD_sys.Ts

    ################# Photo diode definition ################
    DD_sys.sigma_sh = 0
    DD_sys.sigma_th = 1/snr_lin
    DD_sys.responsivity = 1
    DD_sys.on_off_noise = 1

    ################# Channel definition ###################
    alpha_dB_km = 0.2
    beta_2_s2_km = -2.168e-23
    fiber_len_km = 0

    ################# filters creation definition ###################
    rc_alpha = 0
    pulse_shape_len = 3
    channel_filt_len = 1
    rx_filt_len = 1

    g_tx_td = torch.tensor(calc_filters.fd_rc_td(rc_alpha, pulse_shape_len, fs, symbol_time), dtype=torch.cfloat, device=device)
    DD_sys.g_tx_td = g_tx_td[None, None,:]
    channel_td = torch.tensor(calc_filters.CD_fiber_td(alpha_dB_km, beta_2_s2_km, fiber_len_km, channel_filt_len, fs), dtype=torch.cfloat, device=device)
    DD_sys.channel_td = channel_td[None, None,:]
    g_rx_td = torch.tensor(calc_filters.fd_rc_td(0, rx_filt_len, fs, symbol_time/2), dtype=torch.float64, device=device)
    DD_sys.g_rx_td = g_rx_td[None, None,:]

    bits = torch.randint(2,((block_len+sym_mem)*const.m,), device=device)
    info_symbols = const.map(bits)
    tx_syms = const.diff_encoding(info_symbols, init_phase_idx=1)

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
    Psi_e = torch.empty((batch_size, 2*block_len, 2*(block_len+sym_mem)), device=device)
    Psi_o = torch.empty((batch_size, 2*block_len, 2*(block_len+sym_mem)), device=device)
    tx_syms = torch.empty((batch_size,2*(block_len+sym_mem)), device=device)
    
    snr_lin = 10.0 ** ((snr_dB+2*snr_dB_var*(torch.rand(batch_size)-0.5))/10.0)
    for i in range(batch_size):
        y_e[i], y_o[i], Psi_e[i], Psi_o[i], tx_syms[i] = one_batch_data_generation(block_len, sym_mem, snr_lin[i], const, device)

    return y_e, y_o, Psi_e, Psi_o, tx_syms

############################ One Hot functions #################################
def sym_2_oh(mapp_re, mapp_im, syms, device):
    mapp_re_len = len(mapp_re)
    mapp_im_len = len(mapp_im)
    syms_len = syms.size(1)//2
    batch_size = syms.size(0)
    oh_re = torch.eye(mapp_re_len)
    oh_im = torch.eye(mapp_im_len)
    syms_oh_re = torch.empty((batch_size,syms_len,mapp_re_len))
    syms_oh_im = torch.empty((batch_size,syms_len,mapp_im_len))
    for i in range(batch_size):
        for j, (sym_re,sym_im) in enumerate(zip(syms[i,:syms_len],syms[i,syms_len:])):
           syms_oh_re[i,j,:] = oh_re[torch.where(torch.isclose(sym_re,mapp_re,rtol=0, atol=1e-5))]
           syms_oh_im[i,j,:] = oh_im[torch.where(torch.isclose(sym_im,mapp_im,rtol=0, atol=1e-5))]
    syms_oh = torch.cat((syms_oh_re.reshape(batch_size,-1),syms_oh_im.reshape(batch_size,-1)),1)
    syms_oh.to(device)
    return syms_oh

def oh_2_sym(mapp_re, mapp_im, syms_oh, syms_len, device):
    mapp_re_len = len(mapp_re)
    mapp_im_len = len(mapp_im)
    syms_oh_re = syms_oh[:,:syms_len*mapp_re_len].reshape(-1, syms_len, mapp_re_len)
    syms_oh_im = syms_oh[:,syms_len*mapp_re_len:].reshape(-1, syms_len, mapp_im_len)
    syms = torch.cat((torch.matmul(syms_oh_re, mapp_re),torch.matmul(syms_oh_im, mapp_im)),1)
    syms.to(device)
    return syms

############################ differential decoding #################################
def diff_decoding(x, sym_len, device):
    x_re = x[:,:sym_len]
    x_im = x[:,sym_len:]
    u_abs = torch.sqrt(torch.square(x_re)+torch.square(x_im))
    u_phase = torch.abs(torch.diff(torch.atan2(x_im,x_re), prepend=torch.zeros(len(x_re),1, device=device), dim=1))
    return torch.cat((u_abs*torch.cos(u_phase),u_abs*torch.sin(u_phase)),1)

############################### Loss functions ######################################
def per_layer_loss_distance_square(x_oh, x_oh_train, device):
    loss_l = torch.zeros(x_oh.size(0), 1, device=device)        # Denotes the loss in Layer L
    for l, x_oh_l in enumerate(x_oh):
        loss_l[l] = torch.log(torch.Tensor([l+2]).to(device))*torch.mean(torch.mean(torch.square(x_oh_train - x_oh_l),1))
    return loss_l


