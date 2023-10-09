import torch
from torch.nn.functional import conv1d

class DD_system():

    N_sim = None
    N_os = None

    g_tx_td = None
    channel_td = None
    g_rx_td = None
    Ts = None

    G_tx_fd = None
    channel_fd = None
    G_rx_fd = None
    len_fft = None
    
    responsivity = None
    sigma_sh = None
    sigma_th = None
    on_off_noise = None

    def __init__(self, device='cpu'):
        self.device = device    
        pass


    def simulate_system_fd(self, symbols):
        self.check_system_ready_fd_sym(symbols)

        syms_up_samp = self.up_sample_symbols(symbols)

        signal_fd = torch.fft.fft(syms_up_samp, n=self.len_fft)
        signal_fd *= self.G_tx_fd
        signal_fd *= self.channel_fd

        signal_td = torch.fft.ifft(signal_fd, n=self.len_fft)
        signal_td = self.square_law_detection(signal_td)

        signal_fd = torch.fft.fft(signal_td, n=self.len_fft)
        signal_fd *= self.G_rx_fd

        signal_td = torch.fft.ifft(signal_fd, n=self.len_fft)
        delay = int(self.N_sim-1)
        stop = int(delay+symbols.size(1)*self.N_sim)
        return signal_td[:,delay:stop:int(self.N_sim/self.N_os)]


    def simulate_system_td(self, symbols, offset=0):
        self.check_system_ready_td_sym()

        syms_up_samp = self.up_sample_symbols(symbols)
        
        signal =  self.convolve(syms_up_samp, self.g_tx_td)
        signal =  self.convolve(signal, self.channel_td)
        
        signal = self.square_law_detection(signal)

        signal =  self.convolve(signal, self.g_rx_td)
        delay = int(self.N_sim+(self.g_tx_td.size(2)+self.channel_td.size(2)+self.g_rx_td.size(2)-3)/2-1)+offset
        stop = int(delay+symbols.size(1)*self.N_sim)
        return signal[:,delay:stop:int(self.N_sim/self.N_os)]
    
    def get_auxiliary_equiv_channel(self, simbol_memory):
        len = simbol_memory*self.N_os+1
        equiv_channel = self.convolve(self.g_tx_td,self.channel_td)
        delay = int((self.g_tx_td.size(2)+self.channel_td.size(2)-2)/2)
        delta = (len-1)//2
        return equiv_channel[0,0,delay-delta:delay+delta+1] 

    
    def convolve(self, signal, filt):
        filt_len = filt.size(dim=2)
        filt = torch.resolve_conj(torch.flip(filt, [2]))
        return conv1d(signal, filt, padding=filt_len-1)

    def square_law_detection(self,signal):
        abs_signal = torch.abs(signal)
        square_law_signal = self.responsivity*abs_signal**2
        shot_noise = abs_signal * torch.normal(0., self.sigma_sh, size=signal.size(), dtype=torch.float64)
        thermal_noise = torch.normal(0., self.sigma_th, size=signal.size(), dtype=torch.float64)
        return square_law_signal + (shot_noise + thermal_noise)*self.on_off_noise

    def up_sample_symbols(self, symbols):
        return torch.kron(symbols,torch.eye(self.N_sim, device=self.device)[-1])


    def check_system_ready_fd_sym(self,symbols):
        assert self.N_sim is not None, "you must define the property .N_sim (simulation upsampling factor)"
        assert self.N_os is not None, "you must define the property .N_os (system upsampling factor)"
        assert (self.N_sim/self.N_os).is_integer(), "N_sim/self.N_os must be integer"
        assert self.len_fft is not None, "you must defin .len_fft"
        assert self.G_tx_fd is not None, "you must define .G_tx_fd"
        assert len(self.G_tx_fd) == self.len_fft, "the length of .G_tx_fd must be equal to .len_fft"
        assert self.channel_fd is not None, "you must define .channel_fd"
        assert len(self.channel_fd) == self.len_fft, "the length of .channel_fd must be equal to .len_fft"
        assert self.G_rx_fd is not None, "you must define .G_rx_fd"
        assert len(self.G_rx_fd) == self.len_fft, "the length of .G_rx_fd must be equal to .len_fft"
        assert len(symbols)*self.N_sim < self.len_fft, "to many symbols simulated, reduce the symbols or increase the .len_fft"

    def check_system_ready_td_sym(self):
            assert self.N_sim is not None, "you must define the property .N_sim (simulation upsampling factor)"
            assert self.N_os is not None, "you must define the property .N_os (system upsampling factor)"
            assert (self.N_sim/self.N_os).is_integer(), "N_sim/self.N_os must be integer"
            assert self.g_tx_td is not None, "you must define .g_tx_td"
            assert self.channel_td is not None, "you must define .channel_td"
            assert self.g_rx_td is not None, "you must define .g_rx_td"
            assert self.Ts is not None, "you must define .Ts"

    

