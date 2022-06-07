import gc
import sys
import dill
import base64
import argparse
import traceback
import threading
from time import sleep
from torch.utils.data import DataLoader
from xmlrpc.server import SimpleXMLRPCServer

from data import get_dataset

class model:
    def __init__(self, n):
        self.n = n

# saved as config file
status_dict = {}
data_cache = {}

def initialize_worker():
    global data_cache
    global status_dict
    data_cache = {}
    status_dict = {}
    gc.collect()


def preload_data_helper(data_cache, input_path, mode):
    if input_path not in data_cache:
        data_cache[input_path] = get_dataset(input_path, mode)
    return {"message": "Successfully pre-loaded the data..."}

def train_model(data_cache, train_data_path, valid_data_path, model_checkpoint_path, train_fn_string, valid_fn_string, train_config, is_last_worker):
    train_fn = dill.loads(base64.b64decode(train_fn_string))
    print("training model:" + str(model_checkpoint_path) + " on data " + str(train_data_path))
    if train_data_path in data_cache:
        train_dataset = data_cache[train_data_path]
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    else:
        print("data not pre loaded")
        train_dataset = get_dataset(train_data_path, "train")
        data_cache[train_data_path] = train_dataset
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    train_fn(model_checkpoint_path, train_dataloader, train_config)
    if is_last_worker:
        validate_model(data_cache, valid_data_path, model_checkpoint_path, valid_fn_string)
        print("validated model:" + str(model_checkpoint_path) + " on data " + str(valid_data_path))
    return {"message": "Successfully submitted model for training"}

def validate_model(data_cache, valid_data_path, model_checkpoint_path, valid_fn_string):
    valid_fn = dill.loads(base64.b64decode(valid_fn_string))

    if valid_data_path in data_cache:
        valid_dataset = data_cache[valid_data_path]
        valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True)
    
    else:
        print("data not pre loaded")
        valid_dataset = get_dataset(valid_data_path, "valid")
        data_cache[valid_data_path] = valid_dataset
        valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True)
    
    valid_fn(model_checkpoint_path, valid_dataloader)

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

if __name__ == "__main__":
    main()