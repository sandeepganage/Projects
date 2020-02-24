from __future__ import print_function, division
from builtins import range, input

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

# Resize all the images to this
IMAGE_SIZE = [100, 100]

# Training configuration
epochs = 5
batch_size = 32
train_path = 'D://Udemy//Fruit360//fruits-360_dataset//fruits-360//Training'
validation_path = 'D://Udemy//Fruit360//fruits-360_dataset//fruits-360//Test'

# get image files
image_files = glob(train_path+'/*/*.jp*g')
validation_image_files = glob(validation_path+'/*/*.jp*g')

# get total number of classes
folders = glob(train_path+'/*')

# Lets have a look at the images
plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()

# Add preprocessing layer at the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# dont train existing weights. We set trainable attribute to False for each layer
for layer in vgg.layers:
    layer.trainable = False

# Appemd our layers - you can add more layers
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

# create model object
model = Model(inputs=vgg.input, outputs=prediction)

# View the model summary
model.summary()

# model's cost and optimization
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

# instance of ImageDataGenerator
gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

# Test generators to see how it works and some other useful things
test_gen=gen.flow_from_directory(validation_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
