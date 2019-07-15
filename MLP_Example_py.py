from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-  

'''
A MLP network for MNIST digits classification

Modfied from the original code from:  https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist

# load mnist dataset
# Split it into training and test data sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# compute the number of labels
num_labels = len(np.unique(y_train))
print("\n\nThe number of the labels is: ", num_labels)

# convert to one-hot vector, that is length 10 for each y value for each label.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train.shape
# image dimensions (assumed square)
image_size = x_train.shape[1]

print("\n\nOriginal Training image shape = ", image_size)

#In a sequential model could have done this with a keras.layers.Flatten(input_shape=(28,28)) 

input_size = image_size * image_size
print("\n\nTraining Input Size = ", input_size)
print("\n\nSince we are using an MLP, the input must be a 1D tensor",
      "\nSo we multiplied the matrix dimensions by themselves for the image.",
      "\nThe image size was:", image_size, "* ", image_size, "= ", input_size)

# resize and normalize
print(len(x_train[0])) #before, 28 * 28 2D Tensor (a matrix)

x_train = np.reshape(x_train, [-1, input_size]) #28 * 28 to 1D Tensor = 784
print(len(x_train[0])) #Now is a 1D tensor

print((x_train[0])) #before the values were 0 to 255
x_train = x_train.astype('float32') / 255 #scales original values from 0 to 255 to 0.0 to 1.0
print((x_train[0])) #now the values are 0.0 to 1.0


x_test = np.reshape(x_test, [-1, input_size]) # we do the same to the x_test  values.
x_test = x_test.astype('float32') / 255

# network parameters
batch_size = 128
hidden_units = 256 # These are the number of neurons in each MLP layer
dropout = 0.45 # Dropout is a form of regularization, that make neural networks more robust to new unseen test input data
               # Dropout is not used in the output layer and is only used during model training.
               # Dropout is not present when making predictions on test data.

# model is a 3-layer MLP with ReLU and dropout after each layer
model = Sequential()
model.add(Dense(hidden_units, input_dim=input_size)) #The 1st MLP Layer. Only the first layer has input_dim specified as an argument. 
model.add(Activation('relu')) 
model.add(Dropout(dropout))  #First MLP Layer has 256 hidden units, post dropout, (1 - 0.45) * 256 = 140 hidden units participate in layer 2 from 1 
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout)) #Second MLP Layer has 140 hidden units, post dropout, (1 - 0.45) * 140 = 91 hidden units participate in layer 3 from 2 
model.add(Dense(num_labels)) #Then we map to the output length for the number of labels, 10.
# this is the output for one-hot vector
model.add(Activation('softmax')) # softmax squashes the outputs to predicted probabilities of each class that sum to 1.
model.summary()                  # The highest probability class from softmax is the class the model predicts
plot_model(model, to_file='mlp-mnist.png', show_shapes=True)

# loss function for one-hot vector
# use of adam optimizer
# accuracy is good metric for classification tasks
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# train the network
model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

# validate the model on test dataset to determine generalization
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))
