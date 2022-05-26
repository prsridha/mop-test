import gc
import sys
import dill
import base64
import argparse
import traceback
import threading
from time import sleep
from xmlrpc.server import SimpleXMLRPCServer

class model:
    def __init__(self, n):
        self.n = n

# saved as config file
status_dict = {}

def initialize_worker():
    global status_dict
    status_dict = {}
    gc.collect()

def execute(exec_id, code_string, params):
    func = dill.loads(base64.b64decode(code_string))

    def bg_execute(exec_id, func, params):
        try:
            func_result = func(*params)
            status_dict[exec_id] = {"status": "COMPLETED", "result": func_result}
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            sys.stdout.flush()
            status_dict[exec_id] = {"status": "FAILED"}

    status_dict[exec_id] = {"status": "RUNNING"}
    thread = threading.Thread(target=bg_execute, args=(exec_id, func, params,))
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

    server.register_function(execute)
    server.register_function(status)
    server.register_function(initialize_worker)
    server.register_function(is_live)
    server.serve_forever()

if __name__ == "__main__":
    main()