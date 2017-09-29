# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 20:46:35 2017

@author: Stefan Draghici
"""

# Part 1 - Identify frauds with SOM

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
frauds=mappings[(7,8)]
frauds=sc.inverse_transform(frauds)

# Part 2 - Go from unsupervised DL to supervised DL

# Create the matrix of features
customers=dataset.iloc[:, 1:].values

# Create the dependent variable
is_fraud=np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i]=1

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
customers = sc.fit_transform(customers)

# Make the ANN

# Import Keras libs
import keras
from keras.models import Sequential
from keras.layers import Dense

# Init the ANN
classifier=Sequential()

# Add the input layer and first hidden layer with dropout
classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=15))

# Add the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compile the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting classifier to the Training set
classifier.fit(customers, is_fraud, batch_size=1, nb_epoch=5)

# Part 3. Make predictions and evluate the model

# Predicting the Test set results
y_pred = classifier.predict(customers)
y_pred=np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred=y_pred[y_pred[:, 1].argsort()]









