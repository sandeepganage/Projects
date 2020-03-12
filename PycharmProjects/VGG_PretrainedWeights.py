from __future__ import print_function, division
from builtins import range, input

import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Input, Flatten, Lambda, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
from glob import glob

# Re-Size all the images to this
IMAGE_SIZE=[100, 100]

# training config
epochs=5
batch_size=32

train_path='D://Udemy//Fruit360//fruits-360_dataset//fruits-360//Training'
valid_path='D://Udemy//Fruit360//fruits-360_dataset//fruits-360//Test'

image_files=glob(train_path+'/*/*.jp*g')
valid_image_files=glob(valid_path+'/*/*.jp*g')

# get number of classes
folders=glob(train_path+'/*')


# Add pre-processing layer in front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE+[3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

# Add more layer to our network
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

# Create a model object
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

# tell the model what cost and optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['acc']
)

# create an instance of ImageDataGenerator
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

# test generator to see how it works and other useful things
# get label mapping for confusion matrix later
test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels=[None]*len(test_gen.class_indices)
print(labels.__len__())

for k, v in test_gen.class_indices.items():
    labels[v] = k

# create generators
train_generator = gen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size
)

valid_generator = gen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size
)

# fit the model
r = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    steps_per_epoch=len(image_files) // batch_size,
    validation_steps=len(valid_image_files) // batch_size
)

def get_confusion_matrix(data_path, N):
    # we need to see data in the same order
    # for both target and predictions
    print("Generating confusion matrix", N)
    predictions=[]
    targets=[]
    i=0
    for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False,batch_size=batch_size*2):
        i+=1
        if i%50 == 0:
            print(i)
        p=model.predict(x)
        p=np.argmax(p,axis=1)
        y=np.argmax(y,axis=1)
        predictions=np.concatenate((predictions,p))
        targets=np.concatenate((targets, y))
        if len(targets) >= N:
            break

    cm=confusion_matrix(targets, predictions)
    return cm


cm = get_confusion_matrix(train_path, len(image_files))
print(cm)
valid_cm=get_confusion_matrix(valid_path, len(valid_image_files))
print(valid_cm)

import matplotlib.pyplot as plt
# loss
plt.plot(r.history['loss'], label=['train_loss'])
plt.plot(r.history['val_loss'], label=['val_loss'])
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label=['train_acc'])
plt.plot(r.history['val_acc'], label=['val_acc'])
plt.legend()
plt.show()

print("Done!")