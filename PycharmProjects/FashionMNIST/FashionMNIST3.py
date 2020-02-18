from __future__ import print_function, division
from builtins import range

from keras import Sequential, Model
from keras.layers import Activation, Conv2D, Dense, MaxPool2D, Flatten, BatchNormalization, Dropout, Input
from keras.models import model_from_json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def y2Indicator(Y):
    N = len(Y)
    K = len(set(Y))
    I = np.zeros((N, K))
    I[np.arange(N), Y] = 1
    return I


data = pd.read_csv("D://Udemy//FashionMNIST//fashion-mnist_test.csv")
data = data.as_matrix()
np.random.shuffle(data)

X = data[:,1:].reshape(-1, 28, 28, 1) / 255.0
Y = data[:, 0].astype(np.int32)
Y = y2Indicator(Y)

json_file = open("model.json", 'r')
loaded_model_jason = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_jason)

# load weight into new model
loaded_model.load_weights("model.h5")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))



print("Loaded model from disk")