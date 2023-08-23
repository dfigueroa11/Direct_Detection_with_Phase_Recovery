import torch as t
import numpy as np
from itertools import product
"""
BCJR algorithm as an MAP detector, derived using the factor graph framework.
"""

class bcjr_upsamp:
    def __init__(self, taps, EsN0_dB, block_len, constellation, N_os, diff_decoding=False ,device='cpu'):
        assert t.is_tensor(taps)
        self.N_os = N_os
        if taps.dim() == 1: # One channel for all batch elements.
            assert taps.shape[0] > 0
            assert (taps.shape[0]-1)//N_os == (taps.shape[0]-1)/N_os ## integer memory in terms of the symbols
            self.l_sym = (taps.size()[0] - 1)//N_os  # Memory of channel in terms of the symbol
            self.l_ch = taps.size()[0] - 1  # Memory of channel in terms of the samples
            self.batch_size = 1
            self.multiple_channels = False
        elif taps.dim() == 2: # Multiple channels
            self.batch_size = taps.shape[0]
            self.multiple_channels = True
            assert taps.shape[1] > 0
            assert (taps.shape[1]-1)//N_os == (taps.shape[1]-1)/N_os ## integer memory in terms of the symbols
            self.l_sym = (taps.size()[1] - 1)//N_os  # Memory of channel in terms of the symbol
            self.l_ch = taps.size()[1] - 1  # Memory of channel in terms of the samples

        self.h = taps.view((self.batch_size, -1)) # Channel taps (assumed real-valued).
        self.block_len = block_len
        if t.is_tensor(EsN0_dB):
            assert EsN0_dB.shape == self.batch_size # Individual Es/N0 for each batch element
            self.esno_lin = 10 ** (EsN0_dB / 10)
        else:
            self.esno_lin = 10 ** (EsN0_dB / 10) * t.ones((self.batch_size,), device=device)

        self.sigma2 = 1/self.esno_lin
        self.const = constellation
        self.device = device

        self.diff_decodding = diff_decoding

    def compute_true_apps(self, y, log_out, pairwise_beliefs_out=False, P_s0=None):

        ## general considerations: n n/2 2 variable per state
        
        """
        Applies the BCJR algorithm (using BP on a clustered factor graph) to compute the true a posteriori probabilities.
        :param y: (N+L)*N_os receive symbols
        :param log_out: If True, APPs are in log domain, else in linear domain.
        :return: APPs for each symbol in log domain, dim=[batch_size,N,const.M]
        """
        assert self.multiple_channels == False
        assert len(y.shape) == 2
        batch_size = y.shape[0]
        assert y.shape[1] == (self.block_len + self.l_sym) * self.N_os, f"{y.shape[1]} = {(self.block_len + self.l_sym) * self.N_os}"
        assert self.batch_size == 1
        if P_s0 is not None:
            assert t.is_tensor(P_s0)
            assert P_s0.size() == (batch_size, self.const.M)

        channel = self.h[0]

        # Compute all (noise free) rx spaces including oversampling
        tx_spaces_early = [t.tensordot(self.upsamp_select(t.tensor(list(product(self.const.mapping, repeat=(l+2)//2)), device=self.device),0,l,False),t.flip(channel[:l+1].cfloat(), dims=[-1, ]),dims=[[-1, ], [0, ]]) for l in range(self.l_ch)]
        tx_space = [t.tensordot(self.upsamp_select(t.tensor(list(product(self.const.mapping, repeat=self.l_sym+1)), device=self.device),samp,self.l_ch),t.flip(channel[:self.l_ch+1].cfloat(), dims=[-1, ]),dims=[[-1, ], [0, ]]) for samp in range(self.N_os)]
        tx_spaces_late = [t.tensordot(self.upsamp_select(t.tensor(list(product(self.const.mapping, repeat=(l+2)//2)), device=self.device),0,l,True),t.flip(channel[-(l+1):].cfloat(), dims=[-1, ]),dims=[[-1, ], [0, ]]) for l in range(self.l_ch)]
        # Compute msgs from observation node to VN (likelihoods)
        ## repalce (34) (35) (36) and (37) from paper 

        likelihoods = []
        for n in range(self.block_len + self.l_sym):
            if n < self.l_sym:  # The dimension is reduced because we have less than self.l predecessing symbols.
                likelihood = 0
                for samp in range(self.N_os):
                    likelihood += (-self.esno_lin[:,None] * t.abs(y[:, self.N_os*n+samp, None] - t.abs(tx_spaces_early[self.N_os*n+samp].unsqueeze(0))**2) ** 2).view([batch_size] + (n + 1) * [self.const.M])
                likelihoods.append(likelihood)
            elif n >= self.block_len:
                likelihood = 0
                for samp in range(self.N_os):
                    likelihood += (-self.esno_lin[:,None] * t.abs(y[:, self.N_os*n+samp, None].unsqueeze(-1) - t.abs(tx_spaces_late[self.l_ch-(self.N_os*(n-self.block_len)+samp)-1].unsqueeze(0))**2) ** 2).view([batch_size] + (self.block_len + self.l_sym - n) * [self.const.M])
                likelihoods.append(likelihood)
            else:
                likelihood = 0
                for samp in range(self.N_os):
                    likelihood += (-self.esno_lin[:,None] * t.abs(y[:, self.N_os*n+samp, None].unsqueeze(-1) - t.abs(tx_space[samp].unsqueeze(0))**2) ** 2).view([batch_size] + (self.l_sym + 1) * [self.const.M])
                likelihoods.append(likelihood)
            
        # Compute forward and backward path
        f2v_msgs_backward = [-np.log(self.const.M) * t.ones((batch_size, self.const.M), device=self.device)]
        if P_s0 is None:
            f2v_msgs_forward = [-np.log(self.const.M) * t.ones((batch_size, self.const.M), device=self.device)]
        else:
            f2v_msgs_forward = [P_s0]   # useful for differential coding

        for n in range(self.block_len + self.l_sym - 1):
            # VN update
            v2f_forward = likelihoods[n] + f2v_msgs_forward[n]
            n_back = self.block_len + self.l_sym - 1 - n  # inverted index for backward path
            v2f_backward = likelihoods[n_back] + f2v_msgs_backward[n]
            # FN update
            if n < self.l_sym:  # at the beginning the number of dimensions is linearly increasing
                f2v_msgs_forward.append(v2f_forward.unsqueeze(-1).repeat((n + 2) * [1, ] + [self.const.M]))
                f2v_msgs_backward.append(v2f_backward.unsqueeze(1).repeat([1, self.const.M] + (n + 1) * [1, ]))  # dim0 = batches
            elif n >= self.block_len - 1:
                f2v_msgs_forward.append(t.logsumexp(v2f_forward, dim=1))
                f2v_msgs_backward.append(t.logsumexp(v2f_backward, dim=-1))
            else:
                f2v_msgs_forward.append(t.logsumexp(v2f_forward.unsqueeze(-1).repeat((self.l_sym+2) * [1, ] + [self.const.M]), dim=1))
                f2v_msgs_backward.append(t.logsumexp(v2f_backward.unsqueeze(1).repeat([1, self.const.M] + (self.l_sym+1) * [1, ]), dim=-1))
            #normalization
            f2v_msgs_backward[-1] -= t.max(f2v_msgs_backward[-1])
            f2v_msgs_forward[-1] -= t.max(f2v_msgs_forward[-1])
        # Final marginalization.
        
        ## wierd the indexing of n and why likelihood again

        beliefs = t.empty((batch_size, self.block_len, self.const.M), device=self.device)
        for n in range(self.block_len):
            if n == 0:
                beliefs[:, n, :] = f2v_msgs_forward[n] + f2v_msgs_backward[-(n + 1)] + likelihoods[n]
            elif n < self.l_sym:
                beliefs[:, n, :] = t.logsumexp(f2v_msgs_forward[n] + f2v_msgs_backward[-(n + 1)] + likelihoods[n],dim=list(range(1, n + 1)))
            else:
                beliefs[:, n, :] = t.logsumexp(f2v_msgs_forward[n] + f2v_msgs_backward[-(n + 1)] + likelihoods[n],dim=list(range(1, self.l_sym + 1)))
            

        if pairwise_beliefs_out:
            b_ij = t.zeros((batch_size, self.l, self.block_len, self.const.M, self.const.M), device=self.device)
            for l in range(self.l):
                for n in range(l+1, self.block_len):
                    if n == 1:
                        b_ij[:,l,n] = f2v_msgs_forward[n] + f2v_msgs_backward[-(n + 1)] + likelihoods[n]
                    elif n < self.l:
                        b_ij[:,l,n] = t.logsumexp(f2v_msgs_forward[n] + f2v_msgs_backward[-(n + 1)] + likelihoods[n], dim=list(range(1, n-l))+list(range(n-l+1, n+1)))
                    else:
                        b_ij[:,l,n] = t.logsumexp(f2v_msgs_forward[n] + f2v_msgs_backward[-(n + 1)] + likelihoods[n], dim=list(range(1, self.l-l))+list(range(self.l-l+1, self.l+1)))
            # Normalization
            b_ij[:,self.ij_mask] = (b_ij - t.logsumexp(b_ij, dim=(-2,-1))[...,None,None])[:,self.ij_mask]
        
        # Normalization
        beliefs = beliefs - t.logsumexp(beliefs, dim=-1).unsqueeze(-1)
        

        # Return
        if log_out:
            if pairwise_beliefs_out:
                return beliefs, b_ij
            else:
                return beliefs
        else: # Convert to linear domain
            if pairwise_beliefs_out:
                return t.exp(beliefs), t.exp(b_ij)
            else:
                return t.exp(beliefs)

    def compute_true_apps_indiv_channels(self, y, log_out, pairwise_beliefs_out=False):
        """
        Applies the BCJR algorithm (using BP on a clustered factor graph) to compute the true a posteriori probabilities.
        :param y: N+L receive symbols
        :param log_out: If True, APPs are in log domain, else in linear domain.
        :return: APPs for each symbol in log domain, dim=[batch_size,N,const.M]
        """
        assert self.multiple_channels
        assert len(y.shape) == 2
        batch_size = y.shape[0]
        assert y.shape[1] == self.block_len + self.l
        assert self.batch_size == batch_size

        EsN0_lin = self.esno_lin * t.ones(batch_size, device=self.device)


        # Compute all (noise free) rx spaces.
        tx_spaces_early = [t.tensordot(t.flip(self.h[:,:l+1].cfloat(), dims=[-1,]), t.tensor(list(product(self.const.mapping, repeat=l+1)), device=self.device),  dims=[[-1,], [-1,]]) for l in range(self.l)]
        tx_space = t.tensordot(t.flip(self.h[:,:self.l+1].cfloat(), dims=[-1,]), t.tensor(list(product(self.const.mapping, repeat=self.l+1)), device=self.device),dims=[[-1,], [-1,]])
        tx_spaces_late = [t.tensordot(t.flip(self.h[:,-(l+1):].cfloat(), dims=[-1,]), t.tensor(list(product(self.const.mapping, repeat=l+1)), device=self.device),dims=[[-1,], [-1,]]) for l in range(self.l)]

        # Compute msgs from observation node to VN (likelihoods)
        likelihoods = []
        for n in range(self.block_len + self.l):
            if n < self.l:  # The dimension is reduced because we have less than self.l predecessing symbols.
                likelihoods.append((-EsN0_lin[:,None] * t.abs(y[:, n, None] - tx_spaces_early[n]) ** 2).view([batch_size] + (n + 1) * [self.const.M]))
            elif n >= self.block_len:
                likelihoods.append((-EsN0_lin[:,None] * t.abs(y[:, n].unsqueeze(-1) - tx_spaces_late[self.block_len + self.l - 1 - n]) ** 2).view([batch_size] + (self.block_len + self.l - n) * [self.const.M]))
            else:
                likelihoods.append((-EsN0_lin[:,None] * t.abs(y[:, n].unsqueeze(-1) - tx_space) ** 2).view([batch_size] + (self.l + 1) * [self.const.M]))

        # Compute forward and backward path
        f2v_msgs_forward = [-np.log(self.const.M) * t.ones((batch_size, self.const.M), device=self.device)]
        f2v_msgs_backward = [-np.log(self.const.M) * t.ones((batch_size, self.const.M), device=self.device)]
        for n in range(self.block_len + self.l - 1):
            # VN update
            v2f_forward = likelihoods[n] + f2v_msgs_forward[n]
            n_back = self.block_len + self.l - 1 - n  # inverted index for backward path
            v2f_backward = likelihoods[n_back] + f2v_msgs_backward[n]

            # FN update
            if n < self.l:  # at the beginning the number of dimensions is linearly increasing
                f2v_msgs_forward.append(v2f_forward.unsqueeze(-1).repeat((n + 2) * [1, ] + [self.const.M]))
                f2v_msgs_backward.append(v2f_backward.unsqueeze(1).repeat([1, self.const.M] + (n + 1) * [1, ]))  # dim0 = batches
            elif n >= self.block_len - 1:
                f2v_msgs_forward.append(t.logsumexp(v2f_forward, dim=1))
                f2v_msgs_backward.append(t.logsumexp(v2f_backward, dim=-1))
            else:
                f2v_msgs_forward.append(t.logsumexp(v2f_forward.unsqueeze(-1).repeat((self.l+2) * [1, ] + [self.const.M]), dim=1))
                f2v_msgs_backward.append(t.logsumexp(v2f_backward.unsqueeze(1).repeat([1, self.const.M] + (self.l+1) * [1, ]), dim=-1))

        # Final marginalization.
        beliefs = t.empty((batch_size, self.block_len, self.const.M), device=self.device)
        for n in range(self.block_len):
            if n == 0:
                beliefs[:, n, :] = f2v_msgs_forward[n] + f2v_msgs_backward[-(n + 1)] + likelihoods[n]
            elif n < self.l:
                beliefs[:, n, :] = t.logsumexp(f2v_msgs_forward[n] + f2v_msgs_backward[-(n + 1)] + likelihoods[n],dim=list(range(1, n + 1)))
            else:
                beliefs[:, n, :] = t.logsumexp(f2v_msgs_forward[n] + f2v_msgs_backward[-(n + 1)] + likelihoods[n],dim=list(range(1, self.l + 1)))

        if pairwise_beliefs_out:
            b_ij = t.zeros((batch_size, self.l, self.block_len, self.const.M, self.const.M), device=self.device)
            for l in range(self.l):
                for n in range(l+1, self.block_len):
                    if n == 1:
                        b_ij[:,l,n] = f2v_msgs_forward[n] + f2v_msgs_backward[-(n + 1)] + likelihoods[n]
                    elif n < self.l:
                        b_ij[:,l,n] = t.logsumexp(f2v_msgs_forward[n] + f2v_msgs_backward[-(n + 1)] + likelihoods[n], dim=list(range(1, n-l))+list(range(n-l+1, n+1)))
                    else:
                        b_ij[:,l,n] = t.logsumexp(f2v_msgs_forward[n] + f2v_msgs_backward[-(n + 1)] + likelihoods[n], dim=list(range(1, self.l-l))+list(range(self.l-l+1, self.l+1)))

        # Normalization
        beliefs = beliefs - t.logsumexp(beliefs, dim=-1).unsqueeze(-1)
        if pairwise_beliefs_out:
            b_ij[:,self.ij_mask] = (b_ij - t.logsumexp(b_ij, dim=(-2,-1))[...,None,None])[:,self.ij_mask]

        # Return
        if log_out:
            if pairwise_beliefs_out:
                return beliefs, b_ij
            else:
                return beliefs
        else: # Convert to linear domain
            if pairwise_beliefs_out:
                return t.exp(beliefs), t.exp(b_ij)
            else:
                return t.exp(beliefs)

    def upsamp_select(self, symbol_blocks, offset, length, late_states=True):
        if self.diff_decodding:
            symbol_blocks = self.const.diff_encoding(symbol_blocks)
        symbol_blocks = t.kron(symbol_blocks,t.eye(self.N_os)[-1])
        if length >= self.l_ch:
            return symbol_blocks[:,offset:offset+length+1]
        if late_states:
            return symbol_blocks[:,-(length+1):]
        return symbol_blocks[:,:length+1]