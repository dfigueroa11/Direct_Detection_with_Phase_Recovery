import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)

# MIMO channel config
K = 30  # Tranmit antennas.
N = 60  # Receive antennas.
snr = 10.0  # SNR of channel in dB for the training.
snr_lin = 10.0 ** (snr/10.0)  # SNR of channel in lin.

# case VC: Varying channel (randomly generated channel matrix for each new sample).

# Function for training and test data generation according to the FC model of the paper.
def data_generation_VC(K, N, size, snr, snr_var, device):
    t = np.random.randint(2, size=(size, K))*2.0-1 # Random binary TX symbols.
    n = np.random.randn(size, N) # Random noise for each RX antenna.
    r = np.empty(shape=[size, N]) # RX dummy vector
    Hr = np.empty(shape=[size, K])
    HH = np.zeros([size, K, K])
    # Randomly vary the SNR
    current_snr = 10.0 ** (np.random.uniform(low=snr-snr_var,high=snr+snr_var, size=size)/10.0)
    for i in range(size):
        H = np.random.randn(N, K)*1.0/np.sqrt(N)  # Generate random and normalized channel.
        HH[i,:,:] = H.T.dot(H) # Calc (H^T H) as it is used as an input for the neural net.
        # Calc the output vector with additive noise which depends on the current SNR.
        r[i,:] = (H.dot(t[i,:])+n[i,:]*np.sqrt((HH[i,:,:]).trace()/K)/np.sqrt(current_snr[i]))
        Hr[i,:] = H.T.dot(r[i,:]) # Calc (H^T y) as it is used as an input for the neural net.
    t = torch.Tensor(t).to(device)
    Hr = torch.Tensor(Hr).to(device)
    HH = torch.Tensor(HH).to(device)
    return r, Hr, HH, current_snr, t


# DetNet config
layers = 3*K
v_len = 2*K
z_len = 8*K

# Training params
training_steps = 1
batch_size_train = 5
snr_var_train = 3.0 # Maximum absolute deviation of the SNR from its mean in logarithmic scale.

# Test params
test_steps= 1_000
batch_size_test = 5_000
snr_range = np.arange(8, 14, 1)

# Definition of the Loss function
def own_loss(t, t_train, t_ZF):
    loss_l = torch.zeros(len(t), 1, device=device)        # Denotes the loss in Layer L
    for layer in range(1,len(t)+1):
        loss_l[layer-1] = torch.log(torch.Tensor([layer+1]).to(device))*torch.mean(torch.mean(torch.square(t_train - t[layer-1]),1)/torch.mean(torch.square(t_train - t_ZF),1))
    print((torch.mean(torch.square(t_train - t[layer-1]),1)).size())
    print((torch.square(t_train - t[layer-1])).size())
    return loss_l
     

# Definition of the DetNet
class DetNet(nn.Module):
    # Build DetNet
    def __init__(self, layers, K, v_len, z_len):
        # Here we define the trainable parameter (Net)
        super(DetNet, self).__init__()
        # We have to use here nn.ModuleList instead of a PythonList. (Otherwise you’ll get an error saying
        # that your model has no parameters, because PyTorch does not see the parameters of the layers stored
        # in a Python list)
        # Furtheremore, we initialize the linear trafo with normailzed weights
        # Linear Traffos W_1l, W_2l, W_3l
        self.linear_trafo_1_l = nn.ModuleList()
        self.linear_trafo_1_l.extend([nn.Linear(3*K + v_len, z_len) for i in range(1, layers+1)])
        for i in range(0, layers):
            nn.init.normal_(self.linear_trafo_1_l[i].weight, std = 0.01)
            nn.init.normal_(self.linear_trafo_1_l[i].bias, std = 0.01)

        self.linear_trafo_2_l = nn.ModuleList()
        self.linear_trafo_2_l.extend([nn.Linear(z_len, K) for i in range(1, layers+1)])
        for i in range(0, layers):
            nn.init.normal_(self.linear_trafo_2_l[i].weight, std = 0.01)
            nn.init.normal_(self.linear_trafo_2_l[i].bias, std = 0.01)
        
        self.linear_trafo_3_l = nn.ModuleList()
        self.linear_trafo_3_l.extend([nn.Linear(z_len , v_len) for i in range(1, layers+1)])
        for i in range(0, layers):
            nn.init.normal_(self.linear_trafo_3_l[i].weight, std = 0.01)
            nn.init.normal_(self.linear_trafo_3_l[i].bias, std = 0.01)
            
        # For Linear Soft Sign function        
        self.kappa_l = nn.ParameterList()
        self.kappa_l.extend([nn.Parameter(torch.rand(1, requires_grad=True, device=device)) for i in range(1, layers+1)])
        # ReLU as activation faunction
        self.relu = nn.ReLU()
            
    def forward(self, Hr, HH):
        v = torch.zeros(len(Hr), v_len, device=device)   # Internal Memory (state), that is passed to the next layer
        t = torch.zeros(1, len(Hr), K, device=device)    # Transmit vector we want to estimate -> Initalizied as zero
        t_tilde = torch.zeros(len(Hr), K, device=device)    # Transmit vector we want to estimate -> Initalizied as zero
        
        # Send Data through the staced DetNet
        for l in range(1,layers+1):
            # Concatenate the 4 inputs Hy, v, x and HH.
            concat = torch.cat((Hr, v, t[-1,:,:], torch.squeeze(torch.matmul(torch.unsqueeze(t[-1,:,:], 1), HH))), 1)
            
            # Apply linear transformation and rectified linear unit (ReLU).
            z = self.relu(self.linear_trafo_1_l[l-1](concat))               
                
            # Generate new t iterate with a final linear trafo.
            t_tilde = t_tilde + self.linear_trafo_2_l[l-1](z)
            t = torch.cat((t, torch.unsqueeze(-1+self.relu(t_tilde+self.kappa_l[l-1])/torch.abs(self.kappa_l[l-1])-self.relu(t_tilde-self.kappa_l[l-1])/torch.abs(self.kappa_l[l-1]),0)), 0)
            
            # Generate new v iterate with a final linear trafo.
            v = v + self.linear_trafo_3_l[l-1](z)
            del concat, z
            torch.cuda.empty_cache()
        del v, t_tilde
        torch.cuda.empty_cache()
        return t[1:,:,:]
    

model = DetNet(layers, K, v_len, z_len)
model.to(device)

# Adam Optimizer
optimizer = optim.Adam(model.parameters(), eps=1e-07)

results = []
ber = []


for i in range(training_steps):
    # Generate a batch of training data.
    r_train, Hr_train, HH_train, snr_train, t_train = data_generation_VC(K, N, batch_size_train, snr, snr_var_train, device)
    
    # Feed the training data to network and update weights.   
    t = model(Hr_train, HH_train)
    print(t.size)

    # compute loss
    # Calculate optimal decorrelation decoder to normalize the loss function later on.
    t_ZF = torch.squeeze(torch.matmul(torch.unsqueeze(Hr_train,1),torch.inverse(HH_train)), 1)
    loss = torch.sum(own_loss(t, t_train, t_ZF))
    
    # compute gradients
    loss.backward()

    # Adapt weights
    optimizer.step()

    # reset gradients
    optimizer.zero_grad()
    
    # Print the current progress of the training (Loss and BER).
    # Pay attention that we are print the Loss/BER on the Trainings-Dataset.
    # For a real evaulation of the model we should test the model on the test dataset
    if i%500 == 0:       
        results.append(own_loss(t, t_train, t_ZF).detach().cpu().numpy())
        ber.append(1 - torch.mean(t_train.eq(torch.sign(t)).float(),[1,2]).detach().cpu().numpy())
        print('Train step ', i, ', current loss: ', results[-1][-1], ', current ber: ', ber[-1][-1])
    del r_train, Hr_train, HH_train, snr_train, t_train, t
    torch.cuda.empty_cache()
    
# Save the trained model
torch.save(model.state_dict(), 'Det_Net_10')

