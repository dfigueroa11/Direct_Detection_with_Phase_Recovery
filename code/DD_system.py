import torch

class DD_system():

    N_sim = None
    N_os = None
    pulse_shape = None
    beta_2 = None
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
        symbols_up_samp = torch.zeros(int(len(symbols)*self.N_sim), dtype=torch.cfloat)
        symbols_up_samp[::self.N_sim] = symbols
        return torch.functional.conv1d(symbols_up_samp,self.pulse_shape)
    
    def applay_channel(self,symbols):
        signal = torch.zeros(int(len(symbols)*self.N_sim), dtype=torch.cfloat)
        signal[::self.N_sim] = symbols
        n_fft = int(2**torch.ceil(torch.log2(torch.tensor([len(signal)]))))
        pulse_shape_fft = torch.fft.fft(self.pulse_shape, n=n_fft)
        f = torch.arange(0,n_fft)/n_fft*self.Bw
        channel_response_fft = torch.exp(1j*self.beta_2*self.fiber_length*(2*torch.pi*f)**2/2)
        print(len(channel_response_fft))
        print(len(pulse_shape_fft))
        signal_fft = torch.fft.fft(signal, n=n_fft)
        out = torch.fft.ifft(channel_response_fft*signal_fft*pulse_shape_fft, n=n_fft)
        return out[:len(signal)]
    
    def square_law_detection(self,signal):
        abs_signal = torch.abs(signal)
        square_law_signal = self.responsivity*abs_signal**2
        shot_noise = abs_signal * torch.normal(0., self.sigma_sh, size=(1,len(signal)))
        thermal_noise = torch.normal(0., self.sigma_th, size=(1,len(signal)))
        return square_law_signal + (shot_noise + thermal_noise)*self.on_off
    
    

    

