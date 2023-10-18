import torch
import torch.nn as nn

import DetNet_aux_functions as aux_func

class PhaseDetNet(nn.Module):

    def __init__(self, layers, block_len, sym_mem , one_hot_len, v_len, z_len, device):
        super(PhaseDetNet, self).__init__()
        # define the parameters for the linear transformation: (W1,b1), (W2,b2) and (W3,b3)
        sym_len = block_len + sym_mem

        self.linear_trafo_1_l = nn.ModuleList()
        self.linear_trafo_1_l.extend([nn.Linear((sym_len + v_len), z_len) for i in range(layers)]) 
        for i in range(layers):
            nn.init.normal_(self.linear_trafo_1_l[i].weight, std = 0.01)
            nn.init.normal_(self.linear_trafo_1_l[i].bias, std = 0.01)
        
        self.linear_trafo_2_l = nn.ModuleList()
        self.linear_trafo_2_l.extend([nn.Linear(z_len, sym_len*one_hot_len) for i in range(layers)]) 
        for i in range(0, layers):
            nn.init.normal_(self.linear_trafo_2_l[i].weight, std = 0.01)
            nn.init.normal_(self.linear_trafo_2_l[i].bias, std = 0.01)
        
        self.linear_trafo_3_l = nn.ModuleList()
        self.linear_trafo_3_l.extend([nn.Linear(z_len , v_len) for i in range(layers)]) 
        for i in range(0, layers):
            nn.init.normal_(self.linear_trafo_3_l[i].weight, std = 0.01)
            nn.init.normal_(self.linear_trafo_3_l[i].weight, std = 0.01)
        
        # define the parameters for the gradient descent steps: delta_1l, ..., delta_4l
        self.delta1_l = nn.ParameterList()
        self.delta1_l.extend([nn.Parameter(torch.rand(1, requires_grad=True, device=device)) for i in range(layers)])
        
        self.delta2_l = nn.ParameterList()
        self.delta2_l.extend([nn.Parameter(torch.rand(1, requires_grad=True, device=device)) for i in range(layers)])
        
        self.delta3_l = nn.ParameterList()
        self.delta3_l.extend([nn.Parameter(torch.rand(1, requires_grad=True, device=device)) for i in range(layers)])
        
        self.delta4_l = nn.ParameterList()
        self.delta4_l.extend([nn.Parameter(torch.rand(1, requires_grad=True, device=device)) for i in range(layers)])
        
        # ReLU as activation faunction
        self.relu = nn.ReLU()#Hardtanh(min_val=-10, max_val=10)

        # extra internal varaibles
        self.layers = layers
        self.v_len = v_len
        self.sym_len = sym_len
        self.block_len = block_len
        self.sym_mem = sym_mem
        self.one_hot_len = one_hot_len
        self.device = device

    def forward(self, l, x_phase, x_phase_oh, v, x_mag, y_e, y_o, Psi_e, Psi_o, mapp):
        batch_size = y_e.size(0)
        # Send Data through the staced DetNet
        x = torch.cat((x_mag*torch.cos(x_phase),x_mag*torch.sin(x_phase)), dim=-1)
        jacobian = torch.cat((-torch.diag_embed(torch.sin(x_phase)*x_mag),torch.diag_embed(torch.cos(x_phase)*x_mag)), dim=-1)

        Psi_e_x = torch.bmm(Psi_e,x.unsqueeze(-1)).squeeze(-1)
        diag_Psi_e_x = torch.cat((torch.diag_embed(Psi_e_x[:,:self.block_len]),torch.diag_embed(Psi_e_x[:,self.block_len:])),1)
        A_e = torch.bmm(jacobian,torch.bmm(torch.transpose(Psi_e, dim0=1, dim1=2),diag_Psi_e_x))
        
        Psi_o_x = torch.bmm(Psi_o,x.unsqueeze(-1)).squeeze(-1)
        diag_Psi_o_x = torch.cat((torch.diag_embed(Psi_o_x[:,:self.block_len]),torch.diag_embed(Psi_o_x[:,self.block_len:])),1)
        A_o = torch.bmm(jacobian,torch.bmm(torch.transpose(Psi_o, dim0=1, dim1=2),diag_Psi_o_x))

        Psi_e_x_sql = torch.sum(torch.square(Psi_e_x).reshape(batch_size,2,-1),dim=1).unsqueeze(-1)
        Psi_o_x_sql = torch.sum(torch.square(Psi_o_x).reshape(batch_size,2,-1),dim=1).unsqueeze(-1)
        
        q = x_phase - self.delta1_l[l]*torch.bmm(A_e,y_e.unsqueeze(-1)).squeeze(-1) + self.delta2_l[l]*torch.bmm(A_e,Psi_e_x_sql).squeeze(-1) \
            - self.delta2_l[l]*torch.bmm(A_o,y_o.unsqueeze(-1)).squeeze(-1) + self.delta4_l[l]*torch.bmm(A_o,Psi_o_x_sql).squeeze(-1)
            
        # Apply linear transformation and ReLU
        z = self.relu(self.linear_trafo_1_l[l](torch.cat((q, v), 1)))
        # Apply linear transformation
        x_phase_oh += self.linear_trafo_2_l[l](z)
        # proyect and append result
        x_phase = aux_func.oh_2_sym(mapp, x_phase_oh, self.sym_len, self.device).unsqueeze(0)
        # calculate the v for the next layer
        v += self.linear_trafo_3_l[l](z)

        del x, jacobian, Psi_e_x, diag_Psi_e_x, A_e, Psi_o_x, diag_Psi_o_x, A_o, Psi_e_x_sql, Psi_o_x_sql, q, z 
        torch.cuda.empty_cache()

        return x_phase, x_phase_oh, v
                