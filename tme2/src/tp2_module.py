import torch
from torch.utils.tensorboard import SummaryWriter

## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import torch.nn as nn
import datamaestro
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data  import random_split


writer = SummaryWriter()

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float)
datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

class Network(nn.Module): 
    def __init__(self, in_dim, hid, out_dim): 
        super(Network, self).__init__()

        self.lin1 = nn.Linear(in_dim, hid)
        self.tan = nn.Tanh()
        self.lin2 = nn.Linear(hid, out_dim)
        self.mse = nn.MSELoss()

        self.lin1.weight = nn.Parameter(torch.randn((in_dim, hid)).t())
        self.lin1.bias = nn.Parameter(torch.randn(hid))
        self.lin2.weight = nn.Parameter(torch.randn((hid, out_dim)).t())
        self.lin2.bias = nn.Parameter(torch.randn(out_dim))

    def forward(self, input): 
        y1 = self.lin1(input) 
        y2 = self.tan(y1)
        y_pred = self.lin2(y2) 
        return y_pred

writer = SummaryWriter()


hist_loss = []
net = Network(13, 32, 1)
optimizer = optim.Adam(net.parameters(), lr=0.009)


iter = 0
for n_iter in range(iter):
    ##  TODO:  Calcul du forward (loss)
    output = net.forward(datax)
    loss = net.mse(output, datay)
    
    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")
    hist_loss.append(loss.detach().numpy())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()


plt.plot(hist_loss)
plt.show()


# Sequential



liste_module = [nn.Linear(13, 32), nn.Tanh(), nn.Linear(32, 1)]
mse = nn.MSELoss()
hist_loss = []


model = nn.Sequential()
for module in liste_module:
    model.append(module)


optimizer = optim.Adam(model.parameters(), lr=0.009)

iter2 = 100
for n_iter in range(iter2):
    ##  TODO:  Calcul du forward (loss)
    output = model.forward(datax)
    loss = mse(output, datay)
    
    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")
    hist_loss.append(loss.detach().numpy())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()


plt.plot(hist_loss)
plt.show()