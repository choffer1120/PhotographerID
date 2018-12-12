#!/usr/bin/env python
# coding: utf-8

from keras.preprocessing import image
from keras.applications import vgg16, resnet50
import os, math
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
#following added by JMS 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD, Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import itertools
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py
import functools

train_path = "train_separate_24.h5"
val_path = "validate_separate_24.h5"

with h5py.File(train_path, 'r') as hf:
    X_train, y_train = hf['images'][:], hf['labels'][:]
    
with h5py.File(val_path, 'r') as hf:
    X_val, y_val = hf['images'][:], hf['labels'][:]

classes = ["airpixels",
    "fursty",
    "shortstache",
    "cschoonover",
    "thiswildidea",
    "loki",
    "helloemilie",
    "danielkordan",
    "cestmaria",
    "jessfindlay",
    "aaronbhall",
          "adrienneraquel",
          "alexstrohl",
          "CristinaMittermeier",
          "dudelum",
          "edkashi",
          "elliepritts",
          "emmett_sparling",
          "garethpon",
          "haakeaulana",
          "hannes_becker",
          "hirozzzz",
          "jasoncharleshill",
          "kelianne"
          ]

datagen = ImageDataGenerator(zca_whitening=True,zoom_range=.25,horizontal_flip=True)


model = resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(350,350,3), pooling='max')

#unfreeze all layers
for layer in model.layers:
    layer.trainable=True

#make last layer    
last = model.layers[-1].output
x = Dense(len(classes), activation="softmax")(last)

BATCH_SIZE = 32

#set training steps based on the # of images
num_train_samples = X_train.shape[0]
num_valid_samples = X_val.shape[0]
learning_rate = 0.00000001
num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)




epochs = 40
learning_rate = 0.00001
learning_rate_2 = 0.0000001
decay_rate = learning_rate / epochs
momentum = 0.8
adam = Adam(lr=learning_rate, beta_1=0.8, beta_2=0.999, decay=decay_rate, epsilon=0.0000001)
sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)

save_best = keras.callbacks.ModelCheckpoint('best_epoch_res_24User.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
history = keras.callbacks.History()


finetuned_model = Model(model.input, x)
# finetuned_model = model
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'

#First Stage
finetuned_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', top3_acc])
finetuned_model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), steps_per_epoch=len(X_train) / BATCH_SIZE, epochs=epochs, callbacks=[save_best, history], validation_data=(X_val, y_val), shuffle=True)

#Second Stage
adam = Adam(lr=learning_rate_2, beta_1=0.8, beta_2=0.999, decay=learning_rate_2 / epochs, epsilon=0.0000001)
finetuned_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', top3_acc])
finetuned_model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), steps_per_epoch=len(X_train) / BATCH_SIZE, epochs=epochs, callbacks=[save_best, history], validation_data=(X_val, y_val), shuffle=True)


finetuned_model.save('last_epoch_res_24User.h5')
