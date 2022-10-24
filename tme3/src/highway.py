from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path
savepath = Path("model_high.pch")
device = torch.device('cuda' if torch.cuda.is_available() else'cpu')

# Téléchargement des données

from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

class State:
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0, 0

class MonDataset(Dataset):
    def __init__(self, images, labels):

        self.images = torch.tensor(images, dtype=torch.float).reshape(len(images), -1)/255
        self.labels = torch.tensor(labels, dtype=torch.float).reshape(-1, 1)
        
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label
    
    def __len__(self):
        return len(self.images)



class HighWay(nn.Module): 
    def __init__(self, in_dim, n):
        super(HighWay, self).__init__()
        self.n = n
        for i in range(n):
            self.LinH = nn.ModuleList([ nn.Linear(in_dim, in_dim) for j in range(self.n) ])
            self.LinT = nn.ModuleList([ nn.Linear(in_dim, in_dim) for j in range(self.n) ])
            self.sig = nn.Sigmoid()
            self.tan = nn.Tanh()
            self.LinH[i].bias =  nn.Parameter(- 9 * torch.randn(in_dim) - 1 )  # bias wetween (-10 , -1)

    def forward(self, input): 
        for i in range(self.n):
            print("iter",i)
            T = self.sig(self.LinT[i](input))
            H = self.tan(self.LinH[i](input))
            input = H*T + input * (1 - T)

        return input


# Data Loader
train_loader = DataLoader(MonDataset(train_images, train_labels), shuffle=True , batch_size=100)
test_loader = DataLoader(MonDataset(test_images, test_labels), shuffle=True , batch_size=100)

writer = SummaryWriter()
loss = nn.MSELoss()

in_dim = 784
n = 5
epsilon = 0.01

if savepath.is_file():
    print("savapath")
    with savepath.open("rb") as fp:
        state = torch.load(fp) # on recommence depuis le modele sauvegarde
        
else:
    model = HighWay(in_dim, n)
    model = model.to(device)
    optim = optim.Adam(model.parameters(), lr=epsilon)
    state = State(model, optim)


for epoch in range(state.epoch,25):
    print("epoch :",epoch)
    for x,y in train_loader:
        state.optim.zero_grad()
        x = x.to(device)
        xhat = state.model(x)
        l = loss(xhat,y)
        l.backward()
        state.optim.step()
        state.iteration += 1

    writer.add_scalar('Loss/train', l.detach().numpy(), state.epoch)
 
    for x,y in test_loader:
        x = x.to(device)
        y = y.to(device)
        xhat = state.model(x)
        l = loss(xhat,y)


    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save(state,fp)
