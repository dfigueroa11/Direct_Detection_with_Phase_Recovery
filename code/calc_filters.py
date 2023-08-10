import numpy as np

def fd_rc_fd(alpha, len , fs, sym_time):
    f_vec = np.fft.fftfreq(len,1/fs)
    if alpha == 0:
        return np.array(np.abs(f_vec) <= 1/(2*sym_time), dtype=float)
    return np.where(np.abs(f_vec) <= (1-alpha)/(2*sym_time), 1, \
                    1/2*(1+np.cos(np.pi*sym_time/alpha*(np.abs(f_vec)-(1-alpha)/(2*sym_time))))) \
                    *(np.abs(f_vec) <= (1+alpha)/(2*sym_time))

def fd_rc_td(alpha, len, fs, sym_time):
    t_vec = (np.arange(len)-(len-1)/2)/fs
    if alpha == 0:
        return np.sinc(t_vec/sym_time)/sym_time
    return np.where(np.abs(t_vec) == sym_time/(2*alpha), np.pi/(4*sym_time)*np.sinc(1/(2*alpha)), \
                    np.sinc(t_vec/sym_time)/sym_time*(np.cos(np.pi*alpha*t_vec/sym_time))/(1-(2*alpha*t_vec/sym_time)**2))

def CD_fiber_td(alpha_dB_km, beta_2_s2_km, fiber_len_km, len, fs):
    len_fft = np.min([len,2**14])
    fiber_fd = CD_fiber_fd(alpha_dB_km, beta_2_s2_km, fiber_len_km, len_fft, fs)
    fiber_td = np.fft.ifft(fiber_fd, n=len_fft)
    fiber_td = np.concatenate((fiber_td[-(len-1)//2:],fiber_td[0:(len)//2+1]))
    return fiber_td

def CD_fiber_fd(alpha_dB_km, beta_2_s2_km, fiber_len_km, len, fs):
    f_vec = np.fft.fftfreq(len,1/fs)
    alpha = alpha_dB_km / (10 * np.log10(np.exp(1))) # [np/km]
    return np.exp(-alpha*fiber_len_km + 1j*beta_2_s2_km*fiber_len_km*(2*np.pi*f_vec)**2/2)

        