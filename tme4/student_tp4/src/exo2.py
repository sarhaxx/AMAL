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
LATENT_SIZE = 10
#Nb epochs 
Nepochs = 1000

PATH = "../../data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)

data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)

print(len(data_train))
#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

net = RNN(DIM_INPUT, LATENT_SIZE, CLASSES)
optimizer = optim.Adam(net.parameters(), lr=0.001)

writer = SummaryWriter()
Loss = nn.CrossEntropyLoss()

for epoch in range(Nepochs):
    total = []
    for x,y in data_train:
        h = torch.zeros((x.size()[0],LATENT_SIZE))
        x.transpose_(0,1)

        ##  TODO:  Calcul du forward (loss)
        output = net.forward(x,h)
        output = net.decode(output[-1]).softmax(dim=1)
        loss = Loss(output,y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total.append(loss.detach().numpy())

    writer.add_scalar('Loss/train', np.array(total).mean(), epoch)
    print(epoch,np.array(total).mean())


acc = []

for x,y in data_test:
    h = torch.zeros((x.size()[0],LATENT_SIZE))
    x.transpose_(0,1)
    output = torch.argmax(net.decode(net.forward(x,h)[-1]).softmax(dim=1), dim=1)

    y_hat = output.detach().numpy()
    y_tar = y.detach().numpy()
    acc.append(np.where(y_hat == y_tar, 1, 0).sum()/np.size(y_tar))
    

accuracy = np.mean(acc)
print(accuracy)
    
    