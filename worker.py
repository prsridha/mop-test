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

CHECKPOINT_STORAGE_PATH = "/data/cerebro_checkpoint_storage"
CONFIG_STORAGE_PATH = "/data/cerebro_config_storage"

# saved as config file


class CerebroWorker:
    def __init__(self, hostname, port):
        self.status_dict = {}
        self.data_cache = {}

        self.server = SimpleXMLRPCServer((hostname, port), allow_none=True)

        self.server.register_function(self.preload_data_helper)
        self.server.register_function(self.train_model)
        self.server.register_function(self.bg_execute)
        self.server.register_function(self.load_worker_data)
        self.server.register_function(self.train_model_on_worker)
        self.server.register_function(self.status)
        self.server.register_function(self.initialize_worker)
        self.server.register_function(self.is_live)

    def server_forever(self):
        self.server.serve_forever()
    
    def initialize_worker(self):
        self.data_cache = {}
        self.status_dict = {}
        gc.collect()

    def preload_data_helper(self, input_fn_string, input_path):
        input_fn = dill.loads(base64.b64decode(input_fn_string))
        if input_path not in self.data_cache:
            self.data_cache[input_path] = input_fn(input_path)
        logging.info("Successfully pre-loaded the data...")
        return {"message": "Successfully pre-loaded the data..."}

    def train_model(self, input_data_path, model_checkpoint_path, input_fn_string, model_fn_string, train_config):
        input_fn = dill.loads(base64.b64decode(input_fn_string))
        model_fn = dill.loads(base64.b64decode(model_fn_string))
        print("training model:" + str(model_checkpoint_path) + " on data " + str(input_data_path))
        if input_data_path in self.data_cache:
            x_train, y_train = self.data_cache[input_data_path]
        
        else:
            print("data not pre loaded")
            data = input_fn(input_data_path)
            self.data_cache[input_data_path] = data
            x_train, y_train = data

        model_fn(model_checkpoint_path, x_train, y_train, train_config)
        return {"message": "Successfully submitted model for training"}

    def bg_execute(self, exec_id, func, params):
        try:
            func_result = func(*params)
            self.status_dict[exec_id] = {"status": "COMPLETED", "result": func_result}
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            sys.stdout.flush()
            self.status_dict[exec_id] = {"status": "FAILED"}


    def load_worker_data(self, exec_id, params):
        # func = dill.loads(base64.b64decode(code_string))
        self.status_dict[exec_id] = {"status": "RUNNING"}
        thread = threading.Thread(target=self.bg_execute, args=(exec_id, self.preload_data_helper, params,))
        thread.start()
        return base64.b64encode(dill.dumps("LAUNCHED"))

    def train_model_on_worker(self, exec_id, params):
        # func = dill.loads(base64.b64decode(code_string))
        self.status_dict[exec_id] = {"status": "RUNNING"}
        thread = threading.Thread(target=self.bg_execute, args=(exec_id, self.train_model, params,))
        thread.start()
        return base64.b64encode(dill.dumps("LAUNCHED"))


    def status(self, exec_id):
        if exec_id in self.status_dict:
            return base64.b64encode(dill.dumps(self.status_dict[exec_id]))
        else:
            return base64.b64encode(dill.dumps({"status": "INVALID ID"}))

    def is_live(self):
        return True

def main():
    parser = argparse.ArgumentParser(description='Argument parser for generating model predictions.')
    parser.add_argument('--hostname', help='Worker host name', default='0.0.0.0')
    parser.add_argument('--port', help='Worker port', default=7777, type=int)
    args = parser.parse_args()

    print('Starting Cerebro worker on {}:{}'.format(args.hostname, args.port))

    worker = CerebroWorker(args.hostname, args.port)
    worker.server_forever()


if __name__ == '__main__':
    main()