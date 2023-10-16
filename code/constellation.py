#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Code by Luca Schmid (luca.schmid@kit.edu)

import torch as t
import numpy as np

bpsk_mapping = t.tensor([1.0, -1.0], dtype=t.cfloat)
qpsk_mapping = 1/np.sqrt(2) * t.tensor([-1-1j, -1+1j, 1-1j, 1+1j], dtype=t.cfloat)
qam16_mapping = 1/np.sqrt(10) * t.tensor([-3-3j, -3-1j, -3+3j, -3+1j,
                                          -1-3j, -1-1j, -1+3j, -1+1j,
                                          +3-3j, +3-1j, +3+3j, +3+1j,
                                          +1-3j, +1-1j, +1+3j, +1+1j ], dtype=t.cfloat)

"""
Constellation class.
"""

class constellation:
    """
    Class which provides some functions, applied to an arbitrary complex constellation, 
    given in mapping.
    """

    def __init__(self, mapping, device, diff_mapping=None):
        """
        :param mapping: t.Tensor which contains the constellation symbols, sorted according
            to their binary representation (MSB left).
        """
        assert len(mapping.shape) == 1 # mapping should be a 1-dim tensor
        self.mapping = mapping.to(device)
        
        self.M = t.numel(mapping) # Number of constellation symbols.
        self.m = np.log2(self.M).astype(int)
        assert self.m == np.log2(self.M) # Assert that log2(M) is integer
        self.mask = 2 ** t.arange(self.m - 1, -1, -1).to(device)

        self.sub_consts = t.stack([t.stack([t.arange(self.M).reshape(2**(i+1),-1)[::2].flatten(), t.arange(self.M).reshape(2**(i+1),-1)[1::2].flatten()]) for i in range(self.m)]).to(device)
        
        self.phase_list = t.unique(t.round(t.angle(self.mapping), decimals=10))
        self.mag_list = t.unique(t.round(t.abs(self.mapping), decimals=10))
        self.mapping_re = t.unique(t.round(t.real(self.mapping), decimals=10))
        self.mapping_im = t.unique(t.round(t.imag(self.mapping), decimals=10))
        self.diff_mapping = diff_mapping    # 2D tensor with the index of the phase of x_k given U_x (dim=0) and x_k-1 (dim=1)

        self.device = device


    def map(self, bits):
        """
        Maps a given bit_sequence to a sequence of constellation symbols.
        The length of the output sequence is len(bit_sequence) / m.
        The operation is applied to the last axis of bit_sequences.
        bit_sequence is allowed to have other dimensions (e.g. multiple sequences at once)
        as long as the last dimensions is the sequence.
        """
        # Assert that the length of the bit sequence is a multiple of m.
        in_shape = bits.shape
        assert in_shape[-1]/self.m == in_shape[-1]//self.m
        # reshape and convert bits to decimal and use decimal number as index for mapping
        return self.mapping[t.sum(self.mask * bits.reshape(in_shape[:-1] + (-1, self.m)), -1)]
    
    def diff_encoding(self, info_symbols, init_phase_idx=0): 
        """
        applay differential encoding of info_symbols (1D tensor), according to diff_mapping, and
        given a initial state x0 assosiated to the phase index init_phase_idx in diff_mapping. 
        """
        assert self.diff_mapping is not None
        info_sym_phase_idx = t.tensor([t.argmin(t.abs(self.phase_list-angle)) for angle in t.angle(info_symbols)])
        diff_symbols = t.empty_like(info_symbols)
        prev_phase_idx = init_phase_idx
        for i,sym in enumerate(info_symbols):
            diff_symbols[i] = t.abs(sym)*t.exp(1j*self.phase_list[self.diff_mapping[info_sym_phase_idx[i],prev_phase_idx]])
            prev_phase_idx = self.diff_mapping[info_sym_phase_idx[i],prev_phase_idx]
        return diff_symbols

    def bit2symbol_idx(self, bits):
        """
        Returns the symbol number (sorted as in self.mapping) for an incoming sequence of bits.
        This "symbol number" can be used for one-hot representation, for example.
        The length of the output sequence is len(bit_sequence) / m.
        The operation is applied to the last axis of bit_sequences.
        bit_sequence is allowed to have other dimensions (e.g. multiple sequences at once)
        as long as the last dimensions is the sequence.
        """
        # Assert that the length of the bit sequence is a multiple of m.
        in_shape = bits.shape
        assert in_shape[-1]/self.m == in_shape[-1]//self.m
        # reshape and convert bits to decimal and use decimal number as index for mapping
        return t.sum(self.mask * bits.reshape(in_shape[:-1] + (-1, self.m)), -1)

    def demap(self, symbol_idxs):
        """
        Demaps a sequence of constellation symbols, given by their indices in self.mapping, to a sequence of bits.
        The length of the output sequence is len(symbols) * m.
        The operation is applied to the last axis of the input sequence.
        """
        # Assert that the length of the bit sequence is a multiple of m.
        in_shape = symbol_idxs.shape
        # reshape and convert symbol to bits and use decimal number as index for mapping
        return symbol_idxs.unsqueeze(-1).bitwise_and(self.mask).ne(0).view(symbol_idxs.shape[:-1]+(-1,)).float()

    def nearest_neighbor(self, rx_syms):
        """
        Accepts a sequence of (possibly equalized) complex symbols.
        Each sample is hard decided to the constellation symbol, which is nearest (Euclidean distance).
        The output are the idxs of the constellation symbols.
        """
        # Compute distances to all possible symbols.
        distance = t.abs(self.mapping - rx_syms[...,None])
        hard_dec_idx = t.argmin(distance, dim=-1)
        return hard_dec_idx

    def bit_metric_decoder(self, symbol_apps):
        """
        Receives a sequence of symbol APPs. For each symbol, an M-dim tensor indicates the logarithmic
        probability for each of the M possible constellation symbols.
        The bit metric decoder calculates the bit LLRs for each of the m bits for each symbol.
        """
        assert len(symbol_apps.shape) >= 2 # second last: symbol sequence, last: M log APPs
        assert symbol_apps.shape[-1] == self.M

        # For each of the m bits, repartition the M APPs into two subsets regarding the respective bit.
        # The output vector has shape (..., m, 2, M/2).
        subset_probs = t.index_select(symbol_apps,-1, self.sub_consts.flatten()).view(symbol_apps.shape[:-1] + self.sub_consts.shape)
        # Sum up probabilities of all subsets. (exp to go from log to lin domain and log to go back to log domain)
        bitwise_apps = self.jacobian_sum(subset_probs, dim=-1)
        # LLR
        LLR = (bitwise_apps[...,0] - bitwise_apps[...,1]).flatten(start_dim = -2)
        assert symbol_apps.shape[:-2] == LLR.shape[:-1]
        assert symbol_apps.shape[-2]*self.m == LLR.shape[-1]
        assert not t.isinf(LLR).any()
        
        return LLR


    def jacobian_sum(self, msg, dim):
        """
        Computes ln(e^a_1 + e^a_2 + ... e^a_M) of a tensor with last dimension (a_1, a_2, ..., a_M)
        by applying the Jacobian algorithm recursively.
        """
        if msg.shape[dim] == 1:
            return msg.flatten(start_dim=-2)
        else:
            if dim == -1:
                tmp = self.pairwise_jacobian_sum(msg[...,0], msg[...,1])
                for i in range(2, msg.shape[-1]):
                    tmp = self.pairwise_jacobian_sum(tmp, msg[...,i])
            elif dim == -2:
                tmp = self.pairwise_jacobian_sum(msg[...,0,:], msg[...,1,:])
                for i in range(2, msg.shape[-1]):
                    tmp = self.pairwise_jacobian_sum(tmp, msg[...,i,:])
            return tmp


    def pairwise_jacobian_sum(self, msg1, msg2):
        """
        Computes ln(e^msg1 + e^msg2) = max(msg1,msg2) + ln(1+e^-|msg1-msg2|).
        """
        return t.maximum(msg1, msg2) + t.log(1 + t.exp(-t.abs(msg1 - msg2)))

