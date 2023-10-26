from typing import Any
import torch

import MagDetNet
import PhaseDetNet
import DetNet_aux_functions as aux_func

class MagPhaseDetNet():

    angle = None

    def __init__(self, layers, block_len, sym_mem, mapp_mag, mapp_phase, v_len, z_len, device):
        self.layers = layers
        self.block_len = block_len
        self.sym_mem = sym_mem
        self.sym_len = block_len + sym_mem
        self.one_hot_len_mag = len(mapp_mag)
        self.one_hot_len_phase = len(mapp_phase)
        self.v_len = v_len
        self.z_len = z_len
        self.device = device

        self.mag_model = MagDetNet.MagDetNet(layers, block_len, sym_mem, mapp_mag, v_len, z_len, device)
        self.mag_model.to(device)

        self.phase_model = PhaseDetNet.PhaseDetNet(layers, block_len, sym_mem, mapp_phase, v_len, z_len, device)
        self.phase_model.to(device)

    def __call__(self, y_e, y_o, Psi_e, Psi_o, state_mag, state_phase, layers, return_all=True):
        assert layers <= self.layers or layers > 0, f'layers should be between 1 and {self.layers}'
        batch_size = y_e.size(0)
        v_mag = torch.zeros(batch_size, self.v_len, device=self.device)
        x_mag = torch.zeros(1, batch_size, self.sym_len_len, device=self.device)
        x_mag_oh = torch.zeros(batch_size, self.sym_len*self.one_hot_len_mag, device=self.device)
        v_phase = torch.zeros(batch_size, self.v_len, device=self.device)
        x_phase = torch.zeros(1, batch_size, self.sym_len, device=self.device)
        x_phase_oh = torch.zeros(batch_size, self.sym_len*self.one_hot_len_phase, device=self.device)

        for l in range(layers):
            x_mag[l,:,:self.sym_mem] = state_mag
            x_phase[l,:,:self.sym_mem] = state_phase
            x_mag, x_mag_oh, v_mag = self.mag_model(l, x_mag, x_mag_oh, v_mag, x_phase, y_e, y_o, Psi_e, Psi_o)
            x_phase, x_phase_oh, v_phase = self.phase_model(l, x_phase, x_phase_oh, v_phase, x_mag, y_e, y_o, Psi_e, Psi_o)
        
        x_phase = aux_func.phase_correction(x_phase, self.angle, self.device)
        if return_all:
            return x_mag[1:,:,self.sym_mem:], x_phase[1:,:,self.sym_mem:]
        return x_mag[-1,:,self.sym_mem:], x_phase[-1,:,self.sym_mem:]

    def train(self):
        self.mag_model.train()
        self.phase_model.train()

    def eval(self):
        self.mag_model.eval()
        self.phase_model.eval()
