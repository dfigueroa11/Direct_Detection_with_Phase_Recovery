import numpy as np

class DD_system():

    N_sim = None
    N_os = None
    g_tx_td = None
    G_tx_fd = None
    channel_td = None
    channel_fd = None
    g_rx_td = None
    G_rx_fd = None
    len_fft = None
    Ts = None

    responsivity = None
    sigma_sh = None
    sigma_th = None
    on_off_noise = None


    def __init__(self):    
        pass










    

    def simulate_system_fd(self, symbols):
        symbols_up_samp = np.zeros(int(len(symbols)*self.N_sim), dtype=complex)
        symbols_up_samp[self.N_sim-1::self.N_sim] = symbols

        signal_fd = np.fft.fft(symbols_up_samp, n=self.len_fft)
        signal_fd *= self.G_tx_fd
        signal_fd *= self.channel_fd

        signal_td = np.fft.ifft(signal_fd, n=self.len_fft)
        signal_td = self.square_law_detection(signal_td)

        signal_fd = np.fft.fft(signal_td, n=self.len_fft)
        signal_fd *= self.G_rx_fd

        signal_td = np.fft.ifft(signal_fd, n=self.len_fft)
        delay = int(self.N_sim-1)
        stop = int(delay+len(symbols)*self.N_os)
        return signal_td[delay:stop:int(self.N_sim/self.N_os)]

    def simulate_system_td(self, symbols):
        symbols_up_samp = np.zeros(int(len(symbols)*self.N_sim), dtype=complex)
        symbols_up_samp[self.N_sim-1::self.N_sim] = symbols

        signal =  np.convolve(symbols_up_samp, self.g_tx_td)*self.Ts
        signal =  np.convolve(signal, self.channel_td)
        
        signal = self.square_law_detection(signal)

        signal =  np.convolve(signal, self.g_rx_td)*self.Ts
        delay = int(self.N_sim+(len(self.g_tx_td)+len(self.channel_td)+len(self.g_rx_td)-3)/2-1)
        stop = int(delay+len(symbols)*self.N_os)
        return signal[delay:stop:int(self.N_sim/self.N_os)]


    def square_law_detection(self,signal):
        abs_signal = np.abs(signal)
        square_law_signal = self.responsivity*abs_signal**2
        shot_noise = abs_signal * np.random.normal(0, self.sigma_sh, size=(len(signal)))
        thermal_noise = np.random.normal(0, self.sigma_th, size=(len(signal)))
        return square_law_signal + (shot_noise + thermal_noise)*self.on_off_noise
    

    

