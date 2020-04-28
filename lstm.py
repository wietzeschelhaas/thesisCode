# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:30:37 2020

@author: wietz
"""

import pandas
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.optimizers import adam

from sklearn import preprocessing


mnist = tf.keras.datasets.mnist
(x_train,y_train), (x_test, y_test) = mnist.load_data()

#normalize data
x_train = x_train/255.0
x_test_ = x_test/255.0

model = Sequential()

#if going to another reccurent layer put : return_sequences=True
#if going to dense then it wouldnt understadn what a sequence is.
model.add(LSTM(128,input_shape=(28,28), activation="relu", return_sequences=True ))
model.add(Dropout(0.2))

model.add(LSTM(128,activation="relu"))
model.add(Dropout(0.2))

#its normal to have one dense layer before going to output.
model.add(Dense(32,activation="relu"))
model.add(Dropout(0.2))


#for this we have 10 classes
model.add(Dense(10,activation="softmax"))

#decay will decrease lr. So it will stop bouncing around alot, You want to only take bigger learning steps in the beginning
opt= adam(lr=1e-3, decay = 1e-5)

#TODO look this up
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train,y_train, epochs=3, validation_data = (x_test,y_test))