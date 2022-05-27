
import os
import pickle

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from cont import schedule


def input_fn(file_path):
    num_classes = 10
    x_path  = file_path + "/tr_x.pkl"
    y_path = file_path + "/tr_y.pkl"

    with open(x_path, 'rb') as f:
        chunk_x = pickle.load(f)
    with open(y_path, 'rb') as f:
        chunk_y = pickle.load(f)
    
    x_train = x_train.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    return x_train, y_train

def model_fn(model_file, config):
    lr = config["lr"]
    batch_size = config["batch_size"]
    if os.path.isfile(model_file):
        model = tf.keras.models.load_model(model_file)
    else:
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
        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=1)
    model.save(model_file)

def execute_fn(model_checkpoint_path, input_fn_string, model_fn_string, train_fn_string)



ip0 = "http://localhost:7777"
ip1 = "http://localhost:7778"

shard1_path = "/users/vik1497/mop-test/MNIST/dataset/shards/1/"
shard2_path = "/users/vik1497/mop-test/MNIST/dataset/shards/2/"
train_partitions = [shard1_path, shard2_path]

valid_partitions = []

worker_ips = [ip0, ip1]

schedule()