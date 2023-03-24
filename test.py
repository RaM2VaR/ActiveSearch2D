# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:42:28 2022

@author: vinayg
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:25:36 2022

@author: vinayg
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:34:18 2022

@author: vinayg
"""

###############################
##### importing libraries #####
###############################

import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset  
import matplotlib.pyplot as plt 
from scipy.io import savemat
#torch.backends.cudnn.benchmark=True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##### Hyperparameters for federated learning #########
input_size = 784 # 28x28
hidden1_size = 500 
hidden2_size = 200 
num_classes = 10

num_clients = 100
num_selected = 4
num_rounds = 1
num_epochs = 5
batch_size = 100
learning_rate = 0.001




l1_el =input_size
l2_el=hidden1_size

o1=torch.ones(l1_el, 1)
z1= torch.zeros(input_size-l1_el, 1)
p1=torch.cat((o1, z1), 0)
s1=p1[torch.randperm(input_size)]
S1=torch.kron(torch.ones(hidden1_size, 1), s1.T)

o2=torch.ones(l2_el, 1)
z2= torch.zeros(hidden1_size-l2_el, 1)
p2=torch.cat((o2, z2), 0)
s2=p2[torch.randperm(hidden1_size)]
S2=torch.kron(torch.ones(hidden2_size, 1), s2.T)
print(S2)

S2= torch.roll(S2, 1)
print(S2)

# MNIST dataset 
dataset = torchvision.datasets.MNIST(root='./data', 
                                     transform=transforms.ToTensor(), 
                                     download=True)
# Random split
train_set_size = int(len(dataset) * 0.8)
test_set_size = len(dataset) - train_set_size 
train_dataset, test_dataset= torch.utils.data.random_split(dataset, [train_set_size, test_set_size])

# Dividing the training data into num_clients, with each client having equal number of images
train_dataset_split = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)/ num_clients) for _ in range(num_clients)])

# Creating a pytorch loader for a Deep Learning model
train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in train_dataset_split]

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)



examples = iter(test_loader)
example_data, example_targets = examples.next()

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden1_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden1_size, hidden2_size) 
         
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


### Client Update
def client_update(client_model, optimizer, criterion, train_loader, epoch=5):
    """
    This function updates/trains client model on client data
    """
    model.train()
    for e in range(epoch):
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            #data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    return loss.item()


### Global Aggregation
def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    """
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
        global_model.load_state_dict(global_dict)
        for model in client_models:
            model.load_state_dict(global_model.state_dict())
            
def test(global_model, test_loader):
    """This function test the global model on test data and returns test loss and test accuracy """
    model.eval()
    test_loss = 0
    correct = 0
    n_correct = 0
    n_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.reshape(-1, 28*28).to(device), target.to(device)
            output = global_model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()
            _, predicted = torch.max(output.data, 1)
            n_samples += target.size(0)
            n_correct += (predicted == target).sum().item()

    #test_loss /= len(test_loader.dataset)
    acc = n_correct / len(test_loader.dataset)

    #return test_loss, acc
    return acc


global_model = NeuralNet(input_size, hidden1_size, hidden2_size, num_classes).to(device)
client_models_all = [NeuralNet(input_size, hidden1_size, hidden2_size, num_classes).to(device) for _ in range(num_clients)]
client_models = [NeuralNet(input_size, hidden1_size, hidden2_size, num_classes).to(device) for _ in range(num_selected)]


for model in client_models:
    model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global model 

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in client_models]

###### List containing info about learning #########
losses_train = []
losses_test = []
acc_train = []
acc_test = []



# Runnining FL
for r in range(num_rounds):
    # select random clients
    client_idx = np.random.permutation(num_clients)[:num_selected]
    # client update
    loss = 0
    for i in tqdm(range(num_selected)):
        #loss += client_update(client_models[i], optimizer[i], criterion, train_loader[client_idx[i]], epoch=num_epochs)
        #client_models_all[client_idx[i]]=client_models[i]
        #Uniform Selection of entries
        client_models_all[client_idx[i]].l1.weight= nn.Parameter(S1*client_models[i].l1.weight)
        client_models_all[client_idx[i]].l1.bias=client_models[i].l1.bias
        client_models_all[client_idx[i]].relu=client_models[i].relu
        client_models_all[client_idx[i]].l2.weight=nn.Parameter(S2*client_models[i].l2.weight)
        client_models_all[client_idx[i]].l2.bias=client_models[i].l2.bias
        loss += client_update(client_models_all[client_idx[i]], optimizer[i], criterion, train_loader[client_idx[i]], epoch=num_epochs)
        S1= torch.roll(S1, 4)
        S2= torch.roll(S2, 3)
        #client_models[i]=client_models_all[client_idx[i]]
        client_models[i].l1.weight= client_models_all[client_idx[i]].l1.weight
        client_models[i].l1.bias=client_models_all[client_idx[i]].l1.bias
        client_models[i].relu=client_models_all[client_idx[i]].relu
        client_models[i].l2.weight=client_models_all[client_idx[i]].l2.weight
        client_models[i].l2.bias=client_models_all[client_idx[i]].l2.bias
    
    losses_train.append(loss)
    
    # server aggregate
    server_aggregate(global_model, client_models)
    
    acc = test(global_model, test_loader)    
    acc_test.append(acc)
    print('%d-th round' % r)
    print('average train loss %0.3g | test acc: %0.3f' % (loss / num_selected, acc))
    
    
    
mdic = {"acc_test": acc_test, "label": "MNIST_IID"}   
savemat("acc_test_MNIST_IID_SGD.mat", mdic)     
    
    
    
#    for i in range(num_selected):
#        for epoch in range(num_epochs):
#            for batch_idx, (images, labels) in enumerate(train_loader[client_idx[i]]): 
#                images = images.reshape(-1, 28*28).to(device)
#                labels = labels.to(device)
                
#                # Forward pass
#                outputs = client_models[i](images)
#                loss = criterion(outputs, labels)
                
#                # Backward and optimize
#                optimizer[client_idx[i]].zero_grad()
#                loss.backward()
#                optimizer[client_idx[i]].step()
                
        











print("Vinay")