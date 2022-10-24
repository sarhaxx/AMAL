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
ds = prepare_dataset("com.lecun.mnist");
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()


"""# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)

savepath = Path("model.pch")"""

class MonDataset(Dataset):
    def __init__(self, images, labels):

        self.images = images.reshape(len(images), -1)/255
        self.labels = labels
        
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label
    
    def __len__(self):
        return len(self.images)

data = DataLoader ( MonDataset(train_images, train_labels) , shuffle=True , batch_size=1)

for x,y in data:
    #print(x)
    pass


# AutoEncoder 

class AutoEncoder(nn.Module): 
    def __init__(self, in_dim, out_dim):
        super(AutoEncoder, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(out_dim, in_dim)
        self.sig = nn.Sigmoid()
        self.mse = nn.MSELoss()

    def forward(self, input): 
        y1 = self.lin1(input) 
        y2 = self.relu(y1)
        y3 = self.lin2(y2) 
        output = self.sig(y3)
        return output



hist_loss = []

net = AutoEncoder(784, 225)
optimizer = optim.Adam(net.parameters(), lr=0.01)

train_images = torch.tensor(train_images, dtype=torch.float).reshape(len(train_images), -1)/255
output = []

writer = SummaryWriter()

iter = 20
for n_iter in range(iter):
    ##  TODO:  Calcul du forward (loss)
    output = net.forward(train_images)
    loss = net.mse(output,  train_images)
    
    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")
    writer.add_scalar('Loss/train', loss.detach().numpy(), n_iter)
    hist_loss.append(loss.detach().numpy())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

writer1 = SummaryWriter()
output = output.detach().numpy().reshape(60000,28,28)

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer1 = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(output[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer1.add_image(f'samples', images, 0)
savepath = Path("model.pch")