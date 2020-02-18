from keras.utils import np_utils


# Load datasets as train and test
import gzip, sys, pickle
f = gzip.open('D:\\Udemy\MNIST_Letnet\mnist.pkl.gz','rb')

if sys.version_info < (3,):
    data = pickle.load(f)
else:
    data = pickle.load(f, encoding='bytes')
f.close()
(x_train, y_train), (x_test, y_test) = data


# Set type to float32 from uint8
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# Normalize values to 0 & 1
x_train /= 255
x_test /= 255


# Transforms labels to one-hot encoding
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


# Reshape the dataset into 4-D array
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)


# Define LeNet-5 Model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import AveragePooling2D
from keras import models
import keras

# Instantiate an empty model
model = Sequential()
# Add C1 Convolutional layer
model.add(Conv2D(6, kernel_size=(5,5), strides=(1,1), activation='tanh', input_shape=(28,28,1), padding="same"))
# S2 Pooling layer
model.add(AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
# Add C3 Convolutional layer
model.add(Conv2D(16, kernel_size=(5,5), strides=(1,1), activation='tanh', padding="valid"))
# S4 Pooling layer
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Add C5 Fully Connected Convolutional layer
model.add(Conv2D(120, kernel_size=(5,5), strides=(1,1), activation='tanh', padding="valid"))
# Flatten the CNN output so that we can connect to the fully connected layer
model.add(Flatten())
# FC6 Fully Connected Layer
model.add(Dense(10, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
hist = model.fit(x=x_train, y=y_train, epochs=10, batch_size=128, validation_data=(x_test,y_test), verbose=1)


# Evaluate the model
test_score = model.evaluate(x_test, y_test)


# Visualize the Training Process
import matplotlib.pyplot as plt

f,ax = plt.subplots()
ax.plot([None] + hist.history['acc'],'o-')
ax.plot([None] + hist.history['val_acc'],'x-')
# Plot legend and use the best location automatically: loc = 0.
ax.legend(['Train acc', 'Validation acc'], loc = 0)
ax.set_title('Training/Validation acc per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('acc')



import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.plot([None] + hist.history['loss'], 'o-')
ax.plot([None] + hist.history['val_loss'], 'x-')
# Plot legend and use the best location automatically: loc = 0.
ax.legend(['Train Loss', 'Validation Loss'], loc = 0)
ax.set_title('Training/Validation Loss per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')


# Saving model to secondary storage
from keras.models import load_model
model.save('D:\\Udemy\MNIST_Letnet\letNet.h5')
del model
model1 = load_model('D:\\Udemy\MNIST_Letnet\letNet.h5')


# Saving/loading only model's architecture
jason_string = model1.to_json()