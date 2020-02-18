from __future__ import print_function, division
from builtins import range

from keras import Sequential, Model
from keras.layers import Activation, Conv2D, Dense, MaxPool2D, Flatten, BatchNormalization, Dropout, Input

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
i = Input(shape=(28, 28, 1))
x = Conv2D(filters=32, kernel_size=(3,3))(i)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPool2D()(x)

x = Conv2D(filters=64, kernel_size=(3,3))(i)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPool2D()(x)

x = Conv2D(filters=128, kernel_size=(3,3))(i)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPool2D()(x)

x = Flatten()(x)
x = Dense(units=300)(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
x = Dense(units=K)(x)
x = Activation('softmax')(x)

model = Model(inputs=i, outputs=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

r = model.fit(X, Y, validation_split=0.33, epochs=15, batch_size=32)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")

#print("Returned : ",r)

#data_test = pd.read_csv("D://Udemy//FashionMNIST//fashion-mnist_test.csv")
#data_test = data_test.as_matrix()

#X_test = data_test[:,1:].reshape(-1, 28, 28, 1) / 255.0
#Y_test = data_test[:, 0].astype(np.int32)

#Y_test = y2Indicator(Y_test)

#r_test = model.predict(X_test)
