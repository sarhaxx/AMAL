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

# Téléchargement des données

from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

#train_images = torch.tensor(train_images, dtype=torch.float).reshape(len(train_images), -1)/255
#train_labels = torch.tensor(train_labels, dtype=torch.float).reshape(-1,1)


class MonDataset(Dataset):
    def __init__(self, images, labels):

        self.images = torch.tensor(images, dtype=torch.float).reshape(len(images), -1)/255
        self.labels = torch.tensor(labels, dtype=torch.float).reshape(-1,1)
        
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label
    
    def __len__(self):
        return len(self.images)



train_loader = DataLoader ( MonDataset(train_images[:5000], train_labels[:5000]), shuffle=True , batch_size=100)



from pathlib import Path
savepath = Path ("model.pch")
device = torch.device('cuda' if torch.cuda.is_available() else'cpu')



class State:
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0, 0


class AutoEncoder(nn.Module): 
    def __init__(self, in_dim, out_dim):
        super(AutoEncoder, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(out_dim, in_dim)
        self.sig = nn.Sigmoid()
        

    def forward(self, input): 
        y1 = self.lin1(input) 
        y2 = self.relu(y1)
        y3 = self.lin2(y2) 
        output = self.sig(y3)
        return output

writer = SummaryWriter()
loss = nn.MSELoss()
hist_loss = []

if savepath.is_file():
    print("savapath")
    with savepath.open("rb") as fp:
        state = torch.load(fp) # on recommence depuis le modele sauvegarde
        
else:
    print("else")
    model = AutoEncoder(784, 225)
    model = model.to(device)
    optim = optim.Adam(model.parameters(), lr=0.01)
    state = State(model, optim)


for epoch in range(state.epoch,150):
    print(epoch)
    for x,y in train_loader:
        state.optim.zero_grad()
        x = x.to(device)
        xhat = state.model(x)
        l = loss(xhat,x)
        hist_loss.append(l.detach().numpy())
        writer.add_scalar('Loss/train', l.detach().numpy(), state.iteration)
        l.backward()
        state.optim.step()
        state.iteration += 1

    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save(state,fp)
