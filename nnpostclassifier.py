from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.backend import clear_session
import os
import random
#import multiprocessing
import tqdm
import argparse
from datetime import datetime

numParams = 30
FOLDERPATH = './crowd_processed/'

input = tf.keras.layers.Input(shape=numParams) 

x = input

x = tf.keras.layers.Dense(numParams*4, activation='relu')(x)
#x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(numParams*16, activation='relu')(x)
#x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(numParams*2, activation='relu')(x)
label = tf.keras.layers.Dense(1, activation='sigmoid', name='label')(x)

model = tf.keras.models.Model(input, label)

model.compile(optimizer='adam', loss="binary_crossentropy",  metrics=['accuracy'])

X = np.load(FOLDERPATH + "trainX.npy")
Y = np.load(FOLDERPATH + "trainY.npy")

print("Base Accuracy:", 1 - np.sum(Y)/len(Y))

model.fit(X,Y, epochs=80,verbose = 1)

tf.saved_model.save(model, "./savedmodels/nnpost")

clear_session()