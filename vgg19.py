# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:30:17 2018

@author: Dati
"""

import keras

from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from sklearn import preprocessing
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AlphaDropout
from keras.utils import np_utils, to_categorical
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'
#x_train = preprocessing.scale(x_train)
#y_test = preprocessing.scale(x_test)
nb_classes = 10
X_train = x_train.astype('float32')
X_test = x_test.astype('float32')
X_train = X_train / 128 -1
X_test = X_test / 128 -1
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
model = Sequential()

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=((32,32,3))))
model.add(Convolution2D(64, 3, 3, activation='selu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='selu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='selu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='selu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='selu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='selu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='selu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='selu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='selu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='selu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='selu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='selu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='selu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='selu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='selu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='selu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(200, activation='selu'))
model.add(AlphaDropout(0.5))
model.add(Dense(100, activation='selu'))
model.add(AlphaDropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train,
          batch_size=16, nb_epoch=10,
          verbose=1,
          validation_data=(X_test, Y_test))