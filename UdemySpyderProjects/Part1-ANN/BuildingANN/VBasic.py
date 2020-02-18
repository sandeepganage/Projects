import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.examples.tutorials.mnist import input_data

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training ion GPU

data = input_data.read_data_sets('D:\\Udemy\\FashionMNIST\\data\\fashion', one_hot=True)

print("Training Set (images) shape: {shape}".format(shape=data.train.images.shape))
print("Training Set (labels) shape: {shape}".format(shape=data.train.labels.shape))

print("Test Set (images) shape: {shape}".format(shape=data.test.images.shape))
print("Test Set (labels) shape: {shape}".format(shape=data.test.labels.shape))

# Create dictionary of target classes
label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
}

plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(data.train.images[0], (28,28))
curr_lbl = np.argmax(data.train.labels[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(data.test.images[0], (28,28))
curr_lbl = np.argmax(data.test.labels[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

data.train.images[0]