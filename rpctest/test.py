
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
    print("XPATH:" + str(x_path))
    with open(x_path, 'rb') as f:
        chunk_x = pickle.load(f)
    with open(y_path, 'rb') as f:
        chunk_y = pickle.load(f)
    
    x_train = chunk_x.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    y_train = keras.utils.to_categorical(chunk_y, num_classes)
    return x_train, y_train

def model_fn(model_file, x_train, y_train, config):
    lr = config["lr"]
    input_shape = (28, 28, 1)
    num_classes = 10
    # batch_size = config["batch_size"]
    if os.path.isfile(model_file):
        model = tf.keras.models.load_model(model_file)
    else:
        print("new model being built with config:" + str(config))
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
    
    print("calling model fit")
    model.fit(x_train, y_train, batch_size=128, epochs=1)
    model.save(model_file)



ip0 = "http://localhost:7777"
ip1 = "http://localhost:7778"

shard1_path = "/users/vik1497/mop-test/MNIST/dataset/shards/1"
shard2_path = "/users/vik1497/mop-test/MNIST/dataset/shards/2"
train_partitions = [shard1_path, shard2_path]

valid_partitions = []

worker_ips = [ip0, ip1]

param_grid = {
        'lr': [1e-1, 1e-2],
}

param_names = [x for x in param_grid.keys()]

def find_combinations(combinations, p, i):
    """
    :param combinations:
    :param p:
    :param i:
    """
    if i < len(param_names):
        for x in param_grid[param_names[i]]:
            p[param_names[i]] = x
            find_combinations(combinations, p, i + 1)
    else:
        combinations.append(p.copy())

# Grid search. Creating all parameter configuration value sets.
train_configs = []
find_combinations(train_configs, {}, 0)



schedule(worker_ips, train_partitions, valid_partitions, input_fn, model_fn, train_configs, preload_data_to_mem=True)