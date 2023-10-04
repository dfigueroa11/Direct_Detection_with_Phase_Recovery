import torch
import torch.nn as nn

import DetNet_aux_functions as aux_func

class DetNet(nn.Module):

    def __init__(self, layers, sym_len, one_hot_len, v_len, z_len, device):
        super(DetNet, self).__init__()
        # define the parameters for the linear transformation: (W1,b1), (W2,b2) and (W3,b3)
        self.linear_trafo_1_l = nn.ModuleList()
        self.linear_trafo_1_l.extend([nn.Linear(2*(sym_len + v_len), 2*z_len) for i in range(layers)]) #2* because of real and imag part
        for i in range(layers):
            nn.init.normal_(self.linear_trafo_1_l[i].weight, std = 0.01)
            nn.init.normal_(self.linear_trafo_1_l[i].bias, std = 0.01)

        self.linear_trafo_2_l = nn.ModuleList()
        self.linear_trafo_2_l.extend([nn.Linear(2*z_len, sym_len*one_hot_len) for i in range(layers)]) #2* because of real and imag part
        for i in range(0, layers):
            nn.init.normal_(self.linear_trafo_2_l[i].weight, std = 0.01)
            nn.init.normal_(self.linear_trafo_2_l[i].bias, std = 0.01)
        
        self.linear_trafo_3_l = nn.ModuleList()
        self.linear_trafo_3_l.extend([nn.Linear(2*z_len , 2*v_len) for i in range(layers)]) #2* because of real and imag part
        for i in range(0, layers):
            nn.init.normal_(self.linear_trafo_3_l[i].weight, std = 0.01)
            nn.init.normal_(self.linear_trafo_3_l[i].bias, std = 0.01)
            
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
        self.relu = nn.ReLU()

        # extra internal varaibles
        self.layers = layers
        self.v_len = v_len
        self.sym_len = sym_len
        self.one_hot_len = one_hot_len
        self.device = device
            
    def forward(self, y_e, y_o, Psi_e, Psi_o, mapp_re, mapp_im):
        batch_size = y_e.size(0)
        v = torch.zeros(batch_size, self.v_len, device=self.device)
        x = torch.zeros(1, batch_size, self.sym_len, device=self.device)
        x_oh = torch.zeros(1, batch_size, self.sym_len*self.one_hot_len, device=self.device)
        
        # Send Data through the staced DetNet
        for l in range(self.layers):
            # calculate the gradient: q_k
            Psi_e_x = torch.matmul(Psi_e,x[-1])
            diag_Psi_e_x = torch.cat(torch.diag_embed(Psi_e_x[:self.sym_len]),torch.diag_embed(Psi_e_x[self.sym_len:]))
            A_e = torch.matmul(torch.transpose(Psi_e, dim0=1, dim1=2),diag_Psi_e_x)

            Psi_o_x = torch.matmul(Psi_o,x[-1])
            diag_Psi_o_x = torch.cat(torch.diag_embed(Psi_o_x[:self.sym_len]),torch.diag_embed(Psi_o_x[self.sym_len:]))
            A_o = torch.matmul(torch.transpose(Psi_o, dim0=1, dim1=2),diag_Psi_o_x)
            
            q = x[-1] - self.delta1_l[l]*torch.matmul(A_e,y_e) + self.delta2_l[l]*torch.matmul(A_e,torch.square(Psi_e_x)) \
                - self.delta2_l[l]*torch.matmul(A_o,y_o) + self.delta4_l[l]*torch.matmul(A_o,torch.square(Psi_o_x))
            
            # Apply linear transformation and rectified linear unit (ReLU)
            z = self.relu(self.linear_trafo_1_l[l](torch.cat((q, v), 1)))
            # Apply linear transformation
            x_oh = torch.cat((x_oh, x_oh[-1] + self.linear_trafo_2_l[l](z)))
            # proyect and ap
            x = torch.cat((x, aux_func.oh_2_sym(mapp_re , mapp_im, x_oh[-1], self.sym_len, self.device)), 0)
            
            # Generate new v iterate with a final linear trafo.
            v = v + self.linear_trafo_3_l[l](z)
            del Psi_e_x, A_e, diag_Psi_e_x, Psi_o_x, A_o, q, z 
            torch.cuda.empty_cache()
        del v
        torch.cuda.empty_cache()
        return x[1:], x_oh[1:]
    
