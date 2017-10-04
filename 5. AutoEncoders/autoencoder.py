#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 18:12:29 2017

@author: user
"""

# Import libs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Import dataset
movies=pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users=pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings=pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Prepare the training set and test set
training_set=pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set=np.array(training_set, dtype='int')

test_set=pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set=np.array(test_set, dtype='int')

# Get the total num of movies and users
nb_users=int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies=int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Convert the data into an array of users, movies and ratings
def convert(data):
    new_data=[]
    for id_users in range(1, nb_users+1):
        id_movies=data[:,1][data[:,0]==id_users]
        id_ratings=data[:,2][data[:,0]==id_users]
        ratings=np.zeros(nb_movies)
        ratings[id_movies-1]=id_ratings
        new_data.append(list(ratings))
    return new_data

training_set=convert(training_set)
test_set=convert(test_set)

# Convert the data into torch tensors
training_set=torch.FloatTensor(training_set)
test_set=torch.FloatTensor(test_set)

# Create the architecture of the Neural Net
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1=nn.Linear(nb_movies, 20)
        self.fc2=nn.Linear(20, 10)
        self.fc3=nn.Linear(10, 20)
        self.fc4=nn.Linear(20, nb_movies)
        self.activation=nn.Sigmoid()
        
    def forward(self, x):
        x=self.activation(self.fc1(x))
        x=self.activation(self.fc2(x))
        x=self.activation(self.fc3(x))
        x=self.fc4(x)
        return x
    
# Create an instance of autoencoder
sae=SAE()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# Train the Autoencoder
nb_epoch=100

for epoch in range(1, nb_epoch+1):
    train_loss=0
    s=0.0
    
    for id_user in range(nb_users):
        input=Variable(training_set[id_user]).unsqueeze(0)
        target=input.clone()
        
        if torch.sum(target.data>0)>0:
            output=sae(input)
            target.require_grad=False
            output[target==0]=0
            loss=criterion(output, target)
            mean_corrector=nb_movies/float(torch.sum(target.data>0)+1e-10)
            loss.backward()
            train_loss+=np.sqrt(loss.data[0]*mean_corrector)
            s+=1.0
            optimizer.step()
            
    print('epoch: '+str(epoch)+' - loss: '+str(train_loss/s))

# Test the Autoencoder
test_loss=0
s=0.0

for id_user in range(nb_users):
    input=Variable(training_set[id_user]).unsqueeze(0)
    target=Variable(test_set[id_user])
    
    if torch.sum(target.data>0)>0:
        output=sae(input)
        target.require_grad=False
        output[target==0]=0
        loss=criterion(output, target)
        mean_corrector=nb_movies/float(torch.sum(target.data>0)+1e-10)
        test_loss+=np.sqrt(loss.data[0]*mean_corrector)
        s+=1.0
        
print('loss: '+str(test_loss/s))




