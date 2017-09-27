# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
training_set=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=training_set.iloc[:, 1:2].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
training_set=sc.fit_transform(training_set)

# Get the inputs and outputs
X_train=training_set[0:1257]
Y_train=training_set[1:1258]

# Reshape
X_train=np.reshape(X_train, (1257, 1, 1))

# Part 2 - Build the RNN

# Import Keras libs
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Init the RNN
regressor=Sequential()

# Add input layer and LSTM layer
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))