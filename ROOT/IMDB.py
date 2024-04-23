from  keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

max([max(sequence) for sequence in train_data])


word_index = imdb.get_word_index()
reverse_word_index = dict([(val, key) for (key, val) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])

import numpy as np

def vectorize(sequences, dimension=10000): 
		results = np.zeros((len(sequences), dimension))
		for i, sequence in enumerate(sequences):
			results[i, sequence] = 1
		return results

x_train = vectorize(train_data)
x_test = vectorize(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(loss = 'binary_crossentropy',optimizer='rmsprop',  metrics = ['accuracy'])


x_val = x_train[:10000]
y_val = y_train[:10000]

partial_x = x_train[10000:]
partial_y = y_train[10000:]


history = model.fit(partial_x, partial_y, epochs=20, batch_size=512, validation_data=(x_val, y_val))
results = model.evaluate(x_test, y_test)
print(results)









































































from keras.datasets import imdb

# Load the IMDB dataset and split it into training and test sets
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Find the maximum integer index of words in the dataset
max_word_index = max([max(sequence) for sequence in train_data])

# Convert the word index to a dictionary with word indices as keys and words as values
word_index = imdb.get_word_index()
reverse_word_index = dict([(val, key) for (key, val) in word_index.items()])

# Decode the first review in the training set to understand the text data
decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])

import numpy as np

# Function to vectorize sequences into a binary matrix
def vectorize(sequences, dimension=10000): 
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

# Vectorize the training and test data
x_train = vectorize(train_data)
x_test = vectorize(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers

# Define the neural network model architecture
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model with appropriate loss function, optimizer, and evaluation metric
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Split a part of the training data to use as validation data
x_val = x_train[:10000]
y_val = y_train[:10000]
partial_x = x_train[10000:]
partial_y = y_train[10000:]

# Train the model on the partial training data and validate it on the validation data
history = model.fit(partial_x, partial_y, epochs=20, batch_size=512, validation_data=(x_val, y_val))

# Evaluate the model on the test data
results = model.evaluate(x_test, y_test)
print(results)






This code performs sentiment analysis on the IMDb movie review dataset using a neural network model. Let's break down the steps:

Importing Libraries: 
The code starts by importing necessary libraries from Keras and NumPy.
Loading Dataset:
It loads the IMDb dataset using imdb.load_data(), which splits the dataset into training and test sets.
The argument num_words=10000 specifies that only the top 10,000 most frequent words will be kept in the dataset, and other less frequent words will be discarded.
Data Preprocessing:
It finds the maximum integer index of words in the dataset using max([max(sequence) for sequence in train_data]).
The word index is converted into a dictionary where word indices are keys and words are values.
Decoding the Review:
The first review in the training set is decoded back to the original text format using the reverse word index.
Vectorizing Sequences:
The vectorize function converts sequences of integers into a binary matrix representation.
Each row in the binary matrix corresponds to a review, and each column corresponds to a word from the vocabulary. A cell is set to 1 if the word appears in the review, otherwise 0.
Both training and test data are vectorized using this function.
Model Architecture:
A neural network model is defined using the Sequential API from Keras.
The model consists of three fully connected layers with 16 units each, followed by a ReLU activation function.
The input shape for the first layer is specified as (10000,) because the binary matrix representation of each review has 10,000 features (words).
Compiling the Model:
The model is compiled with binary cross-entropy loss function (binary_crossentropy), RMSprop optimizer (rmsprop), and accuracy metric.
Validation Data:
A part of the training data is split to create a validation set for monitoring the model's performance during training.
Training the Model:
The model is trained using the fit method. It is trained on the partial training data and validated on the validation data.
Training is performed for 20 epochs with a batch size of 512.
Model Evaluation:
Finally, the model is evaluated on the test data using the evaluate method.
The evaluation results, including the loss and accuracy, are printed to the console.
Overall, this code demonstrates the complete process of building, training, and evaluating a neural network model for sentiment analysis on the IMDb movie review dataset.







ReLU stands for Rectified Linear Unit, which is an activation function commonly used in artificial neural networks, particularly in deep learning models. The ReLU function is defined as:

f(x) = \max(0, x)f(x)=max(0,x)

In simpler terms, the ReLU function returns 0 for any negative input and returns the input itself for any positive input. Mathematically, it can be represented as a piecewise function:

x & \text{if } x > 0 \\
0 & \text{otherwise}
\end{cases} \]
ReLU has become popular due to its simplicity and effectiveness in training deep neural networks. Some key advantages of ReLU include:
1. **Efficiency**: ReLU is computationally efficient to compute, as it involves simple operations like maximum and comparison.
2. **Sparsity**: ReLU introduces sparsity in the network because neurons output 0 for negative inputs. This sparsity can help in reducing overfitting by preventing the co-adaptation of neurons.
3. **Non-linear**: ReLU introduces non-linearity to the model, allowing it to learn complex relationships in the data.
Despite its advantages, ReLU may suffer from the "dying ReLU" problem, where neurons can become inactive (outputting zero) for all inputs during training, effectively "dying" and becoming unresponsive to gradient updates. This can happen when the neuron's weights are initialized such that the weighted sum of inputs to the ReLU neuron is consistently negative, leading to zero outputs for all inputs. To mitigate this issue, variants of ReLU, such as Leaky ReLU and Parametric ReLU, have been proposed.
