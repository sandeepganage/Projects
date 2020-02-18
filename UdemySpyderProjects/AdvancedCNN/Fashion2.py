from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, BatchNormalization, Flatten

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from keras import callbacks
from keras.callbacks import LearningRateScheduler
from keras.optimizers import adam

log_dir = 'D:/Udemy/FashionMNIST/log2/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

filepath=log_dir+'model--{epoch:02d}-{loss:02f}-{acc:.2f}-{val_loss:02f}-{val_acc:.2f}.h5'

def lr_schedule(epoch):
    """
    Learning rate is scheduled to be reduced after 10, 20, 30, 40 epochs called 
    automatically after every epoch as part of callback during training
    """
    lr = 1e-3
    if epoch > 10:
        lr *= 1e-1
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 30:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-4
    elif epoch > 50:
        lr *= 1e-4
    elif epoch > 60:
        lr *= 1e-4
    elif epoch > 70:
        lr *= 1e-4
    elif epoch > 80:
        lr *= 1e-4
    elif epoch > 90:
        lr *= 1e-4
    elif epoch > 100:
        lr *= 1e-4

    print('Learning rate: ', lr)
    return lr

def y2Indicator(y):
    N = len(y)
    k = len(set(y))
    I = np.zeros((N, k))
    I[np.arange(N), y] = 1
    return I

data = pd.read_csv('D:/Udemy/FashionMNIST/fashion-mnist_test.csv')
data = data.as_matrix()
np.random.shuffle(data)

X = data[:,1:].reshape(-1, 28, 28, 1) / 255.0
Y = data[:, 0].astype(np.int32)

K = len(set(Y))

Y = y2Indicator(Y)

#Callbacks
modelCheck = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=False, mode='auto', period=1)

tensorboardCallback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

lr_Scheduler = LearningRateScheduler(lr_schedule)


from keras.engine.input_layer import Input

# Make the CNN
i = Input(shape=(28,28,1))
x = Conv2D(filters=32, kernel_size=(3,3))(i)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPool2D()(x)

x = Conv2D(filters=64, kernel_size=(3,3))(i)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPool2D()(x)

x = Flatten()(x)
x = Dense(units=100)(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
x = Dense(units=K)(x)
x = Activation('softmax')(x)

model = Model(inputs=i, outputs=x)

model.compile(
        loss='categorical_crossentropy', optimizer='adam', sample_weight_mode='auto', metrics=['accuracy']
        )

r = model.fit(X, Y, validation_split=0.3, epochs=110, batch_size=10, 
              callbacks=[modelCheck, lr_Scheduler, tensorboardCallback])

print(r.history.keys())

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

