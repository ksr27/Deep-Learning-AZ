
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part 1. data preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2. Make the ANN

# Import Keras libs
import keras
from keras.models import Sequential
from keras.layers import Dense

# Init the ANN
classifier=Sequential()

# Add the input layer and first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Add the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Add the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compile the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting classifier to the Training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

# Part 3. Make predictions and evluate the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)

# Predict a single new observation
new_prediction=classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction=(new_prediction>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4. Evaluate and Improve the ANN

# Eval the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)
accuracies=cross_val_score(classifier, X=X_train, y=y_train, cv=10)
mean=accuracies.mean()
variance=accuracies.std()
















