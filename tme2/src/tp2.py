import torch
from torch.utils.tensorboard import SummaryWriter

## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm
import torch.nn.functional as nn
from tp1 import mse, linear
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data  import random_split

writer = SummaryWriter()

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float)
datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

# TODO: 

w = torch.nn.parameter.Parameter(torch.randn((13, 1)))
b = torch.nn.parameter.Parameter(torch.randn(1))


epsilon = 1e-6

hist_loss_train = []
hist_loss_test = []

# data split
train_datax, test_datax = datax[:406] , datax[:-100]
train_datay, test_datay = datay[:406], datay[:-100]


#Batch
#Train
iter = 0

for n_iter in range(iter):
    ##  TODO:  Calcul du forward (loss)
    
    yhat = linear(train_datax, w, b)
    loss = mse(yhat, train_datay)
    
    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    loss.backward()
    hist_loss_train.append(loss.detach().numpy())

    with torch.no_grad():

        w -= w.grad * epsilon
        b -= b.grad * epsilon

        w.grad.zero_()
        b.grad.zero_()
    
    # other 
    """
    w.data -= w.grad.data * epsilon
    b.data -= b.grad.data * epsilon

    # Set the gradients to zero

    w.grad.data.zero_()
    b.grad.data.zero_()

    """

#Test
for n_iter in range(iter):
    ##  TODO:  Calcul du forward (loss)
    
    yhat = linear(test_datax, w, b)
    loss = mse(yhat, test_datay)
    
    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    loss.backward()
    hist_loss_test.append(loss.detach().numpy())

    with torch.no_grad():
        w -= w.grad * epsilon
        b -= b.grad * epsilon

        # Set the gradients to zero
        w.grad.zero_()
        b.grad.zero_()

plt.plot(hist_loss_train)
plt.plot(hist_loss_test)
plt.show()


# Mini Batch 

epsilon = 1e-9
iter2 = 50000
for n_iter in range(iter2):
    yhat = linear(datax, w, b)
    loss = mse(yhat, datay)
    
    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    loss.backward()
    hist_loss_test.append(loss.detach().numpy())
    # Retropropagation
    if n_iter % 100 == 0:
        with torch.no_grad():
            w -= w.grad * epsilon
            b -= b.grad * epsilon

            # Set the gradients to zero
            w.grad.zero_()
            b.grad.zero_()


plt.plot(hist_loss_train)
plt.plot(hist_loss_test)
plt.show()