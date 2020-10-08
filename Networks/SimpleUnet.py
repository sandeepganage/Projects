import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

K.set_image_data_format('channels_first')

global flag
flag = True


def unet(width, height, depth, classes, weightsPath=None):
    global flag

    data_shape = width * height
    input_shape = (depth, height, width)
    input_img = Input(shape=input_shape)

#    inputs = Input(input_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
    conv1_BN1 = BatchNormalization(axis=1)(conv1, training=flag)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_BN1)
    conv1_BN2 = BatchNormalization(axis=1)(conv1, training=flag)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_BN2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2_BN1 = BatchNormalization(axis=1)(conv2, training=flag)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_BN1)
    conv2_BN2 = BatchNormalization(axis=1)(conv2, training=flag)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_BN2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3_BN1 = BatchNormalization(axis=1)(conv3, training=flag)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_BN1)
    conv3_BN2 = BatchNormalization(axis=1)(conv3, training=flag)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_BN2)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4_BN1 = BatchNormalization(axis=1)(conv4, training=flag)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_BN1)
    conv4_BN2 = BatchNormalization(axis=1)(conv4, training=flag)
    drop4 = Dropout(0.5)(conv4_BN2)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5_BN1 = BatchNormalization(axis=1)(conv5, training=flag)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5_BN1)
    conv5_BN2 = BatchNormalization(axis=1)(conv5, training=flag)
    drop5 = Dropout(0.5)(conv5_BN2)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))

    merge6 = concatenate([drop4, up6], axis=1)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6_BN1 = BatchNormalization(axis=1)(conv6, training=flag)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6_BN1)
    conv6_BN2 = BatchNormalization(axis=1)(conv6, training=flag)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6_BN2))

    merge7 = concatenate([conv3_BN2, up7], axis=1)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7_BN1 = BatchNormalization(axis=1)(conv7, training=flag)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7_BN1)
    conv7_BN2 = BatchNormalization(axis=1)(conv7, training=flag)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7_BN2))
    merge8 = concatenate([conv2_BN2, up8], axis=1)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8_BN1 = BatchNormalization(axis=1)(conv8, training=flag)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8_BN1)
    conv8_BN2 = BatchNormalization(axis=1)(conv8, training=flag)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8_BN2))
    merge9 = concatenate([conv1, up9], axis=1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9_BN1 = BatchNormalization(axis=1)(conv9, training=flag)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_BN1)
    conv9_BN2 = BatchNormalization(axis=1)(conv9, training=flag)
    conv9 = Conv2D(classes, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_BN2)
#    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    reshape = Reshape((classes, data_shape), input_shape=(classes, height, width))(conv9)
    permut = Permute((2, 1))(reshape)
    softmax = Activation('softmax', dtype="float32")(permut)
    model = keras.Model(inputs=[input_img], outputs=[softmax])

    model.summary()
    if weightsPath is not None:
        model.load_weights(weightsPath)
        print("pre-trained weights loaded successfully")
    return model

"""
    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model
"""

unet(512, 512, 3, 5, weightsPath=None)