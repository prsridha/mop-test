#!/usr/bin/env python
# coding: utf-8

import os
import pickle
from mnist import MNIST

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def load_data(path):
    mndata = MNIST(path)
    tr_x, tr_y = mndata.load_training()
    tst_x, tst_y = mndata.load_testing()
    print("Training:", len(tr_x))
    print("Testing:", len(tst_x))
    return (tr_x, tr_y, tst_x, tst_y)


def save_train_shards(n, tr_x, tr_y):
    total = len(tr_x)
    chunk = total//n
    
    for i in range(n):
        os.mkdir('./dataset/shards/' + str(i+1))
        chunk_x = tr_x[chunk*i:(chunk)*(i+1)]
        chunk_y = tr_y[chunk*i:(chunk)*(i+1)]
        
        chunk_x = np.asarray(chunk_x).reshape(chunk, 28, 28)
        chunk_y = np.asarray(chunk_y)
        
        with open('./dataset/shards/' + str(i+1) + "/tr_x.pkl", 'wb') as f:
            pickle.dump(chunk_x, f)
        with open('./dataset/shards/' + str(i+1) + "/tr_y.pkl", 'wb') as f:
            pickle.dump(chunk_y, f)


def read_train(n):
    for i in range(n):
        with open('./dataset/shards/' + str(i+1) + "/tr_x.pkl", 'rb') as f:
            chunk_x = pickle.load(f)
        with open('./dataset/shards/' + str(i+1) + "/tr_y.pkl", 'rb') as f:
            chunk_y = pickle.load(f)
             
        return chunk_x, chunk_y

def model():
    num_classes = 10
    input_shape = (28, 28, 1)
    
    x_train, y_train = read_train(2)
    x_train = x_train.astype("float32") / 255
    
    x_train = np.expand_dims(x_train, -1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    print(model.summary())
    
    batch_size = 128
    epochs = 15

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    # pickle model here and then load it via NFS.
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)



tr_x, tr_y, tst_x, tst_y = load_data("dataset")

save_train_shards(3, tr_x, tr_y)

# model()

# read_train(2)
