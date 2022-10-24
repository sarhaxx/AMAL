from utils import RNN, device,SampleMetroDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 1
#Taille du batch
BATCH_SIZE = 32

#Taille Latent
LATENT_SIZE = 20
#Nb epochs 
Nepochs = 10

PATH = "../../data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)

data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)

print(len(data_train))
#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

net = RNN(DIM_INPUT, LATENT_SIZE, CLASSES)
optimizer = optim.Adam(net.parameters(), lr=0.01)

writer = SummaryWriter()
Loss = nn.CrossEntropyLoss()

h = torch.zeros((2,LATENT_SIZE))


for epoch in range(Nepochs):
    total=0.
    for x,y in data_train:
        print(x.size())
        print(h.size())
        print(y.size())
        ##  TODO:  Calcul du forward (loss)
        output = net.forward(x,h)
        loss = Loss(output,y)
        # Sortie directe
        #print(f"Itérations {n_iter}: loss {loss}")
        writer.add_scalar('Loss/train', loss.detach().numpy(), n_iter)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total+=loss

    print(epoch,total)