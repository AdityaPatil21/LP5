from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 

fashion_mnist = keras.datasets.fashion_mnist
(train_img, train_labels), (test_img, test_labels) = fashion_mnist.load_data()

train_img = train_img / 255.0
test_img = test_img / 255.0

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), 
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'])

model.fit(train_img, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_img, test_labels)
print("accuracy of tessting: ",test_acc)

predictions = model.predict(test_img)

predicted_labels = np.argmax(predictions, axis=1)

num_rows = 5 
num_cols = 5
num_imgs = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_imgs):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plt.imshow(test_img[1], cmap='gray')
    plt.axis("off")
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plt.bar(range(10), predictions[i])
    plt.xticks(range(10))
    plt.ylim([0,1])
    plt.tight_layout()
    plt.title(f"predicted_labels: {predicted_labels[i]}")
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# Importing necessary libraries
from tensorflow import keras  # TensorFlow's Keras API for building neural networks
import numpy as np  # Library for numerical computations
import matplotlib.pyplot as plt  # Library for creating visualizations

# Loading the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_img, train_labels), (test_img, test_labels) = fashion_mnist.load_data()

# Preprocessing: Scaling pixel values to the range [0, 1]
train_img = train_img / 255.0
test_img = test_img / 255.0

# Defining the neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Flatten layer to convert 2D images into 1D arrays
    keras.layers.Dense(128, activation='relu'),  # Fully connected layer with 128 neurons and ReLU activation
    keras.layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each class) and softmax activation
])

# Compiling the model with optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model on the training data for 10 epochs
model.fit(train_img, train_labels, epochs=10)

# Evaluating the model on the test data
test_loss, test_acc = model.evaluate(test_img, test_labels)
print("Accuracy of testing:", test_acc)

# Making predictions on the test data
predictions = model.predict(test_img)

# Extracting predicted labels from probability distributions
predicted_labels = np.argmax(predictions, axis=1)

# Plotting a grid of images with their corresponding predicted labels and probability distributions
num_rows = 5 
num_cols = 5
num_imgs = num_rows * num_cols

plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_imgs):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plt.imshow(test_img[1], cmap='gray')
    plt.axis("off")
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plt.bar(range(10), predictions[i])
    plt.xticks(range(10))
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.title(f"Predicted Label: {predicted_labels[i]}")
    plt.show()







This code loads the Fashion MNIST dataset, preprocesses it, defines a neural network model, trains the model, evaluates its performance, makes predictions, and visualizes the results. Let's go through each part:

Importing Libraries: TensorFlow's Keras API is imported for building neural networks, along with NumPy for numerical computations and Matplotlib for visualization.
Loading Data: The Fashion MNIST dataset is loaded using Keras. It consists of grayscale images of clothing items (28x28 pixels) belonging to 10 different categories.
Data Preprocessing: Pixel values of the images are scaled to the range [0, 1] by dividing by 255.0, which standardizes the data.
Model Definition: A sequential neural network model is defined. It consists of a Flatten layer to convert 2D images into 1D arrays, a Dense hidden layer with 128 neurons and ReLU activation, and a Dense output layer with 10 neurons and softmax activation.
Model Compilation: The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy as the evaluation metric.
Model Training: The model is trained on the training data for 10 epochs using the fit method.
Model Evaluation: The trained model is evaluated on the test data using the evaluate method to compute the test loss and accuracy.
Prediction: Predictions are made on the test data using the predict method, obtaining probability distributions over the 10 classes.
Visualization: A grid of images along with their predicted labels and probability distributions is plotted using Matplotlib for visual inspection.







This code snippet demonstrates the process of building, training, and evaluating a neural network model for the Fashion MNIST dataset using TensorFlow and Keras. Let's break it down step by step:

Import Libraries:
tensorflow.keras: TensorFlow's high-level neural networks API, Keras, is imported to facilitate building and training the model.
numpy: A library for numerical computations in Python, used here for array manipulation.
matplotlib.pyplot: A plotting library for creating visualizations in Python.
Load Dataset:
The Fashion MNIST dataset is loaded using the keras.datasets.fashion_mnist module. This dataset consists of grayscale images of clothing items belonging to 10 different categories.
The dataset is split into training and testing sets, each comprising images and corresponding labels.
Data Preprocessing:
Pixel values of the images are scaled to the range [0, 1] by dividing them by 255.0. This standardizes the data, making it easier for the neural network to learn.
Define the Model:
The neural network model is defined using the Sequential API provided by Keras. It's a sequential stack of layers.
The first layer is a Flatten layer that converts the 2D image data into a 1D array.
Two fully connected (Dense) layers follow. The first one has 128 neurons with ReLU activation, and the second one has 10 neurons with softmax activation. The softmax layer outputs probability scores for each of the 10 classes.
Compile the Model:
The model is compiled with the adam optimizer, sparse_categorical_crossentropy loss function, and accuracy as the evaluation metric.
Model Training:
The model is trained using the fit method on the training data (train_img and train_labels) for 10 epochs.
Model Evaluation:
The model's performance is evaluated on the test data (test_img and test_labels) using the evaluate method. The test loss and accuracy are computed and printed.
Prediction and Visualization:
Predictions are made on the test images using the predict method, which returns the probability scores for each class.
Predicted labels are obtained by selecting the class with the highest probability score using argmax.
A grid of images with their corresponding predicted labels and probability distributions is plotted using Matplotlib.
This code showcases a complete pipeline for training and evaluating a simple neural network model for image classification using TensorFlow and Keras, along with visualizing the results.
