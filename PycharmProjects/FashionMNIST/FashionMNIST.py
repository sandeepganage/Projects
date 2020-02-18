from __future__ import print_function, division
from builtins import range

from keras import Sequential
from keras.layers import Activation, Conv2D, Dense, MaxPool2D, Flatten, BatchNormalization, Dropout

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def y2Indicator(Y):
    N = len(Y)
    K = len(set(Y))
    I = np.zeros((N, K))
    I[np.arange(N), Y] = 1
    return I


data = pd.read_csv("D://Udemy//FashionMNIST//fashion-mnist_train.csv")
data = data.as_matrix()
np.random.shuffle(data)

X = data[:,1:].reshape(-1, 28, 28, 1) / 255.0
Y = data[:, 0].astype(np.int32)

# Finding total number of labels
K = len(set(Y))

Y = y2Indicator(Y)

# Model building
model = Sequential()
model.add(Conv2D(input_shape=(28,28,1), filters=32, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D())

model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D())

model.add(Conv2D(filters=128, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(units=300))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(units=K))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

r = model.fit(X, Y, validation_split=0.33, epochs=15, batch_size=32)
print("Returned : ",r)


print("Hello")

