import numpy as np

class DD_system():

    N_sim = None
    N_os = None
    pulse_shape = None
    beta_2 = None
    fiber_alpha = None 
    fiber_length = None
    Ts = None
    symbol_time = None
    symbol_rate = None
    Bw = None
    responsivity = None
    sigma_sh = None
    sigma_th = None
    rng = None
    on_off = None


    def __init__(self):    
        pass


    def pulse_shaping(self,symbols):
        symbols_up_samp = np.zeros(int(len(symbols)*self.N_sim), dtype=complex)
        symbols_up_samp[::self.N_sim] = symbols
        return np.convolve(symbols_up_samp,self.pulse_shape)
    
    def applay_channel(self,signal):
        n_fft = int(16*2**np.ceil(np.log2(len(signal))))
        f = self.Bw * np.fft.fftfreq(n_fft)
        alpha_lin = self.fiber_alpha / (10 * np.log10(np.exp(1)))
        channel_response_fft = np.exp(-alpha_lin*self.fiber_length+1j*self.beta_2*self.fiber_length*(2*np.pi*f)**2/2)
        signal_fft = np.fft.fft(signal, n=n_fft)
        out = np.fft.ifft(channel_response_fft*signal_fft, n=n_fft)
        return out[:len(signal)]
    
    def square_law_detection(self,signal):
        abs_signal = np.abs(signal)
        square_law_signal = self.responsivity*abs_signal**2
        shot_noise = abs_signal * np.random.normal(0, self.sigma_sh, size=(len(signal)))
        thermal_noise = np.random.normal(0, self.sigma_th, size=(len(signal)))
        return square_law_signal + (shot_noise + thermal_noise)*self.on_off
    
    

    

