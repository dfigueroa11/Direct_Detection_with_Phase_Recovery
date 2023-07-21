import numpy as np

def fd_rc_fd(alpha, len , fs, sym_time):
    f_vec = np.fft.fftfreq(len,1/fs)
    if alpha == 0:
        return np.array(np.abs(f_vec) <= 1/(2*sym_time), dtype=float)
    return np.where(np.abs(f_vec) <= (1-alpha)/(2*sym_time), 1, 1/2*(1+np.cos(np.pi*sym_time/alpha*(np.abs(f_vec)-(1-alpha)/(2*sym_time)))))*(np.abs(f_vec) <= (1+alpha)/(2*sym_time))

def CD_fiber_fd(alpha_dB_km, beta_2_s2_km, fiber_len_km, len, fs):
    f_vec = np.fft.fftfreq(len,1/fs)
    alpha = alpha_dB_km / (10 * np.log10(np.exp(1))) # [np/km]
    return np.exp(-alpha*fiber_len_km + 1j*beta_2_s2_km*fiber_len_km*(2*np.pi*f_vec)**2/2)

        