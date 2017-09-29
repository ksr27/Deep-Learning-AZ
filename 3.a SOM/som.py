# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:50:44 2017

@author: Stefan Draghici
"""

import numpy as np
import pandas as pd

# Import the dataset

dataset=pd.read_csv('Credit_Card_Applications.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler

sc=MinMaxScaler(feature_range=(0, 1))
X=sc.fit_transform(X)

# Train the SOM
from minisom import MiniSom

som=MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Visualize results
from pylab import bone, pcolor, colorbar, plot, show

bone()
pcolor(som.distance_map().T)
colorbar()
markers=['o', 's']
colors=['r', 'g']

for i, x in enumerate(X):
    w=som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

# Find the frauds
mappings=som.win_map(X)
frauds=mappings[(8,2)]
frauds=sc.inverse_transform(frauds)