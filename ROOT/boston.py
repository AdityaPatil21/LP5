import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn import preprocessing

(X_train, Y_train), (X_test, Y_test) = keras.datasets.boston_housing.load_data()

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
print("Train output data shape:", Y_train.shape)
print("Actual Test output data shape:", Y_test.shape)

##Normalize the data

X_train=preprocessing.normalize(X_train)
X_test=preprocessing.normalize(X_test)

#Model Building

X_train[0].shape
model = Sequential()
model.add(Dense(128,activation='relu',input_shape= X_train[0].shape))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse',optimizer='rmsprop',metrics=['mae'])

history = model.fit(X_train,Y_train,epochs=100,batch_size=1,verbose=1,validation_data=(X_test,Y_test))

results = model.evaluate(X_test, Y_test)
print(results)




















import numpy as np  # Library for numerical computations
import tensorflow as tf  # TensorFlow library
from tensorflow import keras  # TensorFlow's Keras API
from tensorflow.keras import Sequential  # Sequential model type for building neural networks
from tensorflow.keras.layers import Dense, Flatten  # Layers for the neural network architecture
from sklearn import preprocessing  # Library for data preprocessing

# Load the Boston Housing dataset
(X_train, Y_train), (X_test, Y_test) = keras.datasets.boston_housing.load_data()

# Display the shapes of training and test data
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
print("Train output data shape:", Y_train.shape)
print("Actual Test output data shape:", Y_test.shape)

##Normalize the data

# Normalize the training and test data
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)

# Model Building

# Check the shape of a single training data instance
X_train[0].shape

# Define the neural network architecture
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=X_train[0].shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer with one neuron (for regression task)

# Display the summary of the model architecture
model.summary()

# Compile the model with appropriate loss function, optimizer, and evaluation metric
model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

# Train the model on the training data
history = model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=1, validation_data=(X_test, Y_test))

# Evaluate the trained model on the test data
results = model.evaluate(X_test, Y_test)
print("Test loss and Mean Absolute Error (MAE):", results)



This code performs the following tasks:

Importing Libraries: Necessary libraries are imported including NumPy for numerical computations, TensorFlow and its Keras API for building and training neural networks, and scikit-learn for data preprocessing.
Loading Data: The Boston Housing dataset is loaded using Keras. It consists of housing-related features and target prices.
Data Exploration: The shapes of the training and test data arrays are printed to understand the dataset's dimensions.
Data Normalization: The input features are normalized using preprocessing.normalize() to scale them to a standard range.
Model Building: A Sequential model is defined. It consists of multiple Dense layers, where the first layer has 128 neurons and ReLU activation function, followed by two hidden layers with 64 and 32 neurons respectively, also using ReLU activation, and an output layer with one neuron (for regression) without activation.
Model Compilation: The model is compiled with Mean Squared Error (MSE) as the loss function, RMSprop optimizer, and Mean Absolute Error (MAE) as the evaluation metric.
Model Training: The model is trained on the training data for 100 epochs using a batch size of 1. The training progress is printed during training, and validation data is provided to monitor the model's performance.
Model Evaluation: The trained model is evaluated on the test data to compute the test loss and MAE.







This code performs the following tasks:

Importing Libraries: Necessary libraries are imported including NumPy for numerical computations, TensorFlow and its Keras API for building and training neural networks, and scikit-learn for data preprocessing.
Loading Data: The Boston Housing dataset is loaded using Keras. It consists of housing-related features and target prices. The data is split into training and test sets (X_train, Y_train), (X_test, Y_test).
Data Exploration: The shapes of the training and test data arrays are printed to understand the dataset's dimensions.
Data Normalization: The input features are normalized using preprocessing.normalize() to scale them to a standard range.
Model Building: A Sequential model is defined. It consists of multiple Dense layers, where the first layer has 128 neurons and ReLU activation function, followed by two hidden layers with 64 and 32 neurons respectively, also using ReLU activation, and an output layer with one neuron (for regression) without activation.
Model Summary: The summary of the model architecture is printed, displaying the layers, their output shapes, and the number of parameters.
Model Compilation: The model is compiled with Mean Squared Error (MSE) as the loss function, RMSprop optimizer, and Mean Absolute Error (MAE) as the evaluation metric.
Model Training: The model is trained on the normalized training data for 100 epochs using a batch size of 1. The training progress is printed during training, and validation data is provided to monitor the model's performance.
Model Evaluation: The trained model is evaluated on the normalized test data to compute the test loss and MAE. The evaluation results are printed.

