import torch
import torch.nn as nn

import DetNet_aux_functions as aux_func

class DetNet(nn.Module):

    def __init__(self, layers, block_len, sym_mem , one_hot_len_mag, one_hot_len_phase, v_len, z_len, device):
        super(DetNet, self).__init__()
        # define the parameters for the linear transformation: (W1,b1), (W2,b2) and (W3,b3)
        sym_len = block_len + sym_mem

        self.linear_trafo_1_l_mag = nn.ModuleList()
        self.linear_trafo_1_l_phase = nn.ModuleList()
        self.linear_trafo_1_l_mag.extend([nn.Linear((sym_len + v_len), z_len) for i in range(layers)]) 
        self.linear_trafo_1_l_phase.extend([nn.Linear((sym_len + v_len), z_len) for i in range(layers)]) 
        for i in range(layers):
            nn.init.normal_(self.linear_trafo_1_l_mag[i].weight, std = 0.01)
            nn.init.normal_(self.linear_trafo_1_l_mag[i].bias, std = 0.01)
            nn.init.normal_(self.linear_trafo_1_l_phase[i].weight, std = 0.01)
            nn.init.normal_(self.linear_trafo_1_l_phase[i].bias, std = 0.01)
        self.linear_trafo_1_l = [self.linear_trafo_1_l_mag, self.linear_trafo_1_l_phase]

        self.linear_trafo_2_l_mag = nn.ModuleList()
        self.linear_trafo_2_l_phase = nn.ModuleList()
        self.linear_trafo_2_l_mag.extend([nn.Linear(z_len, sym_len*one_hot_len_mag) for i in range(layers)]) 
        self.linear_trafo_2_l_phase.extend([nn.Linear(z_len, sym_len*one_hot_len_phase) for i in range(layers)]) 
        for i in range(0, layers):
            nn.init.normal_(self.linear_trafo_2_l_mag[i].weight, std = 0.01)
            nn.init.normal_(self.linear_trafo_2_l_mag[i].bias, std = 0.01)
            nn.init.normal_(self.linear_trafo_2_l_phase[i].weight, std = 0.01)
            nn.init.normal_(self.linear_trafo_2_l_phase[i].bias, std = 0.01)
        self.linear_trafo_2_l = [self.linear_trafo_2_l_mag, self.linear_trafo_2_l_phase]

        self.linear_trafo_3_l_mag = nn.ModuleList()
        self.linear_trafo_3_l_phase = nn.ModuleList()
        self.linear_trafo_3_l_mag.extend([nn.Linear(z_len , v_len) for i in range(layers)]) 
        self.linear_trafo_3_l_phase.extend([nn.Linear(z_len , v_len) for i in range(layers)]) 
        for i in range(0, layers):
            nn.init.normal_(self.linear_trafo_3_l_mag[i].weight, std = 0.01)
            nn.init.normal_(self.linear_trafo_3_l_mag[i].weight, std = 0.01)
            nn.init.normal_(self.linear_trafo_3_l_phase[i].bias, std = 0.01)
            nn.init.normal_(self.linear_trafo_3_l_phase[i].bias, std = 0.01)
        self.linear_trafo_3_l = [self.linear_trafo_3_l_mag, self.linear_trafo_3_l_phase]
    
        # define the parameters for the gradient descent steps: delta_1l, ..., delta_4l
        self.delta1_l_mag = nn.ParameterList()
        self.delta1_l_phase = nn.ParameterList()
        self.delta1_l_mag.extend([nn.Parameter(torch.rand(1, requires_grad=True, device=device)) for i in range(layers)])
        self.delta1_l_phase.extend([nn.Parameter(torch.rand(1, requires_grad=True, device=device)) for i in range(layers)])
        self.delta1_l = [self.delta1_l_mag, self.delta1_l_phase]
        
        self.delta2_l_mag = nn.ParameterList()
        self.delta2_l_phase = nn.ParameterList()
        self.delta2_l_mag.extend([nn.Parameter(torch.rand(1, requires_grad=True, device=device)) for i in range(layers)])
        self.delta2_l_phase.extend([nn.Parameter(torch.rand(1, requires_grad=True, device=device)) for i in range(layers)])
        self.delta2_l = [self.delta2_l_mag, self.delta2_l_phase]

        self.delta3_l_mag = nn.ParameterList()
        self.delta3_l_phase = nn.ParameterList()
        self.delta3_l_mag.extend([nn.Parameter(torch.rand(1, requires_grad=True, device=device)) for i in range(layers)])
        self.delta3_l_phase.extend([nn.Parameter(torch.rand(1, requires_grad=True, device=device)) for i in range(layers)])
        self.delta3_l = [self.delta3_l_mag, self.delta3_l_phase]

        self.delta4_l_mag = nn.ParameterList()
        self.delta4_l_phase = nn.ParameterList()
        self.delta4_l_mag.extend([nn.Parameter(torch.rand(1, requires_grad=True, device=device)) for i in range(layers)])
        self.delta4_l_phase.extend([nn.Parameter(torch.rand(1, requires_grad=True, device=device)) for i in range(layers)])
        self.delta4_l = [self.delta4_l_mag, self.delta4_l_phase]

        # ReLU as activation faunction
        self.relu = nn.Hardtanh(min_val=-10, max_val=10)

        # extra internal varaibles
        self.layers = layers
        self.v_len = v_len
        self.sym_len = sym_len
        self.block_len = block_len
        self.sym_mem = sym_mem
        self.one_hot_len_mag = one_hot_len_mag
        self.one_hot_len_phase = one_hot_len_phase
        self.device = device

    def forward(self, y_e, y_o, Psi_e, Psi_o, mapp_mag, mapp_phase):
        batch_size = y_e.size(0)
        v_mag = torch.zeros(batch_size, self.v_len, device=self.device)
        x_mag = torch.zeros(1, batch_size, self.sym_len, device=self.device)
        x_mag_oh = torch.zeros(batch_size, self.sym_len*self.one_hot_len_mag, device=self.device)
        v_phase = torch.zeros(batch_size, self.v_len, device=self.device)
        x_phase = torch.zeros(1, batch_size, self.sym_len, device=self.device)
        x_phase_oh = torch.zeros(batch_size, self.sym_len*self.one_hot_len_phase, device=self.device)

        # Send Data through the staced DetNet
        for l in range(self.layers):
            # make the magnitude stage
            jacobian = torch.cat((torch.diag_embed(torch.cos(x_phase[-1])),torch.diag_embed(torch.sin(x_phase[-1]))), dim=-1)
            x_mag, x_mag_oh, v_mag = self.gradient(y_e, y_o, Psi_e, Psi_o, x_mag, x_phase, jacobian, mapp_mag, l, x_mag_oh, v_mag, stage=0)
            # make the phase stage
            jacobian = torch.cat((-torch.diag_embed(torch.sin(x_phase[-1])*x_mag[-1]),torch.diag_embed(torch.cos(x_phase[-1])*x_mag[-1])), dim=-1)
            x_phase, x_phase_oh, v_phase = self.gradient(y_e, y_o, Psi_e, Psi_o, x_mag, x_phase, jacobian, mapp_phase, l, x_phase_oh, v_phase, stage=1)
    
            
        del jacobian, x_mag_oh, v_mag, x_phase_oh, v_phase
        torch.cuda.empty_cache()

        return x_mag[1:], x_phase[1:]

    def gradient(self, y_e, y_o, Psi_e, Psi_o, x_mag, x_phase, jacobian, mapp, l, x_oh, v, stage):
        batch_size = y_e.size(0)
        x = torch.cat((x_mag[-1]*torch.cos(x_phase[-1]),x_mag[-1]*torch.sin(x_phase[-1])), dim=-1)
        Psi_e_x = torch.bmm(Psi_e,x.unsqueeze(-1)).squeeze(-1)
        diag_Psi_e_x = torch.cat((torch.diag_embed(Psi_e_x[:,:self.block_len]),torch.diag_embed(Psi_e_x[:,self.block_len:])),1)
        A_e = torch.bmm(jacobian,torch.bmm(torch.transpose(Psi_e, dim0=1, dim1=2),diag_Psi_e_x))
        
        Psi_o_x = torch.bmm(Psi_o,x.unsqueeze(-1)).squeeze(-1)
        diag_Psi_o_x = torch.cat((torch.diag_embed(Psi_o_x[:,:self.block_len]),torch.diag_embed(Psi_o_x[:,self.block_len:])),1)
        A_o = torch.bmm(jacobian,torch.bmm(torch.transpose(Psi_o, dim0=1, dim1=2),diag_Psi_o_x))

        Psi_e_x_sql = torch.sum(torch.square(Psi_e_x).reshape(batch_size,2,-1),dim=1).unsqueeze(-1)
        Psi_o_x_sql = torch.sum(torch.square(Psi_o_x).reshape(batch_size,2,-1),dim=1).unsqueeze(-1)
        
        if stage == 0:
            q = x_mag[-1] - self.delta1_l[stage][l]*torch.bmm(A_e,y_e.unsqueeze(-1)).squeeze(-1) + self.delta2_l[stage][l]*torch.bmm(A_e,Psi_e_x_sql).squeeze(-1) \
                - self.delta2_l[stage][l]*torch.bmm(A_o,y_o.unsqueeze(-1)).squeeze(-1) + self.delta4_l[stage][l]*torch.bmm(A_o,Psi_o_x_sql).squeeze(-1)
        elif stage == 1:
            q = x_phase[-1] - self.delta1_l[stage][l]*torch.bmm(A_e,y_e.unsqueeze(-1)).squeeze(-1) + self.delta2_l[stage][l]*torch.bmm(A_e,Psi_e_x_sql).squeeze(-1) \
                - self.delta2_l[stage][l]*torch.bmm(A_o,y_o.unsqueeze(-1)).squeeze(-1) + self.delta4_l[stage][l]*torch.bmm(A_o,Psi_o_x_sql).squeeze(-1)
            
        # Apply linear transformation and ReLU
        z = self.relu(self.linear_trafo_1_l[stage][l](torch.cat((q, v), 1)))
        # Apply linear transformation
        x_oh = x_oh + self.linear_trafo_2_l[stage][l](z)
        # proyect and append result
        if stage == 0:
            x_mag = torch.cat((x_mag, aux_func.oh_2_sym(mapp, x_oh, self.sym_len, self.device).unsqueeze(0)), 0)
        elif stage == 1:
            x_phase = torch.cat((x_phase, aux_func.oh_2_sym(mapp, x_oh, self.sym_len, self.device).unsqueeze(0)), 0)
        # Generate new v iterate with a final linear trafo.
        v = v + self.linear_trafo_3_l[stage][l](z)

        del x, Psi_e_x, diag_Psi_e_x, A_e, Psi_o_x, diag_Psi_o_x, A_o, Psi_e_x_sql, Psi_o_x_sql, q, z 
        torch.cuda.empty_cache()

        if stage == 0:
            return x_mag, x_oh, v
        elif stage == 1:
            return x_phase, x_oh, v

        return -1
