import os
import sys
import gc
import dill
import argparse
import base64
import threading
import logging
import traceback

import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from xmlrpc.server import SimpleXMLRPCServer

dataset = None

logging.basicConfig(filename="worker.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

namespace = 'cerebro'

CHECKPOINT_STORAGE_PATH = "./cerebro_checkpoint_storage/"
CONFIG_STORAGE_PATH = "./cerebro_config_storage/"
DATA_STORAGE_PATH = "./cerebro_data_storage/"

# saved as config file
status_dict = {}
data_cache = {}

def initialize_worker():
    global data_cache
    global status_dict
    data_cache = {}
    status_dict = {}
    gc.collect()

def preload_data_helper(data_cache, input_fn_string, input_path):
    input_fn = dill.loads(base64.b64decode(input_fn_string))
    if input_path not in data_cache:
        data_cache[input_path] = input_fn(input_path)
    logging.info("Successfully pre-loaded the data...")
    return {"message": "Successfully pre-loaded the data..."}

def train_model(data_cache, input_data_path, model_checkpoint_path, input_fn_string, model_fn_string, train_config):
    input_fn = dill.loads(base64.b64decode(input_fn_string))
    model_fn = dill.loads(base64.b64decode(model_fn_string))
    print("training model:" + str(model_checkpoint_path) + " on data " + str(input_data_path))
    if input_data_path in data_cache:
        x_train, y_train = data_cache[input_data_path]
    
    else:
        print("data not pre loaded")
        data = input_fn(input_data_path)
        data_cache[input_data_path] = data
        x_train, y_train = data

    model_fn(model_checkpoint_path, x_train, y_train, train_config)
    return {"message": "Successfully submitted model for training"}

def bg_execute(exec_id, func, params):
    try:
        func_result = func(data_cache, *params)
        status_dict[exec_id] = {"status": "COMPLETED", "result": func_result}
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        sys.stdout.flush()
        status_dict[exec_id] = {"status": "FAILED"}


def load_worker_data(exec_id, params):
    # func = dill.loads(base64.b64decode(code_string))
    status_dict[exec_id] = {"status": "RUNNING"}
    thread = threading.Thread(target=bg_execute, args=(exec_id, preload_data_helper, params,))
    thread.start()
    return base64.b64encode(dill.dumps("LAUNCHED"))

def train_model_on_worker(exec_id, params):
    # func = dill.loads(base64.b64decode(code_string))
    status_dict[exec_id] = {"status": "RUNNING"}
    thread = threading.Thread(target=bg_execute, args=(exec_id, train_model, params,))
    thread.start()
    return base64.b64encode(dill.dumps("LAUNCHED"))


def status(exec_id):
    if exec_id in status_dict:
        return base64.b64encode(dill.dumps(status_dict[exec_id]))
    else:
        return base64.b64encode(dill.dumps({"status": "INVALID ID"}))

def is_live():
    return True

def main():
    parser = argparse.ArgumentParser(description='Argument parser for generating model predictions.')
    parser.add_argument('--hostname', help='Worker host name', default='0.0.0.0')
    parser.add_argument('--port', help='Worker port', default=7777, type=int)
    args = parser.parse_args()

    print('Starting Cerebro worker on {}:{}'.format(args.hostname, args.port))
    server = SimpleXMLRPCServer((args.hostname, args.port), allow_none=True)


    server.register_function(preload_data_helper)
    server.register_function(train_model)
    server.register_function(bg_execute)
    server.register_function(load_worker_data)
    server.register_function(train_model_on_worker)
    server.register_function(status)
    server.register_function(initialize_worker)
    server.register_function(is_live)
    server.serve_forever()

if __name__ == '__main__':
    main()