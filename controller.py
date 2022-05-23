import os
import json
import logging
import numpy as np
from flask import Flask, request

app = Flask(__name__)
logging.basicConfig(filename="controller.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

namespace = 'cerebro'

CHECKPOINT_STORAGE_PATH = "./cerebro_checkpoint_storage/"
CONFIG_STORAGE_PATH = "./properties/"
HYPERPARAM_STORAGE_PATH = "./hyperparameter_properties/"
WORKER_ENDPOINT = "http://cerebro-service-worker-{}:6000"
OPTIONAL_HYPERPARAMS = ['batch_size']
REQUIRED_HYPERPARAMS = ['learning_rate', 'lambda_value']


@app.route("/health")
def health():
    return "Hi this is controller. I'm working!\n"

def get_num_epochs():
    # config.load_incluster_config()
    # api_instance = client.CoreV1Api()
    # r = api_instance.read_namespaced_config_map(name, namespace, pretty=False)
    properties_path = os.path.join(
        CONFIG_STORAGE_PATH, "cerebro-properties.json")
    properties = json.loads(properties_path)
    num_epochs = properties["num_epochs"]
    return num_epochs


def get_param_grid():
    # config.load_incluster_config()
    # api_instance = client.CoreV1Api()
    # r = api_instance.read_namespaced_config_map(name, namespace, pretty=False)
    properties_path = os.path.join(CONFIG_STORAGE_PATH, "cerebro-properties.json")
    properties = json.loads(properties_path)
    param_grid = properties["param_grid"]
    return param_grid


def get_num_models():
    param_grid = get_param_grid()
    config_sizes = [len(param_grid[key]) for key in param_grid]
    return np.prod(config_sizes)


def get_worker_states():
    path = os.path.join(CONFIG_STORAGE_PATH, "worker-states.json")
    f = open(path, "r")
    worker_states_json = json.loads(f.read())
    worker_states = {}
    for key in worker_states_json:
        worker_states[int(key)] = worker_states_json[key]
    f.close()
    return worker_states


def set_worker_states(worker_states):
    path = os.path.join(CONFIG_STORAGE_PATH, "worker-states.json")
    f = open(path, "w")
    f.write(json.dumps(worker_states))
    f.close()


def get_epoch_number():
    # config.load_incluster_config()
    # api_instance = client.CoreV1Api()
    # r = api_instance.read_namespaced_config_map(name, namespace, pretty=False)
    properties_path = os.path.join(
        CONFIG_STORAGE_PATH, "cerebro-properties.json")
    properties = json.loads(properties_path)
    epoch_number = properties["current_epoch"]
    return epoch_number


def set_epoch_number(epoch_number):
    # config.load_incluster_config()
    # api_instance = client.CoreV1Api()
    # cfm = api_instance.read_namespaced_config_map(name, namespace, pretty=False)
    properties_path = os.path.join(
      CONFIG_STORAGE_PATH, "cerebro-properties.json")
    properties = json.loads(properties_path)
    properties["current_epoch"] = epoch_number
    with open(properties_path, "w") as f:
        f.write(json.dumps(properties))
    # api_instance.patch_namespaced_config_map(
    #     name, namespace, body=cfm, pretty=True)


def get_model_states():
    path = os.path.join(CONFIG_STORAGE_PATH, "model-states.json")
    f = open(path, "r")
    model_states_json = json.loads(f.read())
    model_states = {}
    for key in model_states_json:
        model_states[int(key)] = model_states_json[key]
    f.close()
    return model_states


def set_model_states(model_states):
    path = os.path.join(CONFIG_STORAGE_PATH, "model-states.json")
    f = open(path, "w")
    f.write(json.dumps(model_states))
    f.close()


def get_model_worker_pairs():
    path = os.path.join(CONFIG_STORAGE_PATH, "model-worker-pairs.json")
    f = open(path, "r")
    model_worker_pairs = json.loads(f.read())
    f.close()
    return model_worker_pairs


def set_model_worker_pairs(model_worker_pairs):
    path = os.path.join(CONFIG_STORAGE_PATH, "model-worker-pairs.json")
    f = open(path, "w")
    f.write(json.dumps(model_worker_pairs))
    f.close()


def get_model_on_worker():
    path = os.path.join(CONFIG_STORAGE_PATH, "model-on-worker.json")
    f = open(path, "r")
    model_on_worker = json.loads(f.read())
    f.close()
    return model_on_worker


def set_model_on_worker(model_on_worker):
    path = os.path.join(CONFIG_STORAGE_PATH, "model-on-worker.json")
    f = open(path, "w")
    f.write(json.dumps(model_on_worker))
    f.close()
    

def get_runnable_model(worker_id, is_train=True):
    model_worker_pairs = get_model_worker_pairs()
    model_states = get_model_states()

    for m, w in model_worker_pairs:
        if w == worker_id and ((is_train and not model_states[m]) or not is_train):
            return m
    return -1


def get_num_workers():
    # config.load_incluster_config()
    # api_instance = client.CoreV1Api()
    # r = api_instance.read_namespaced_config_map(name, namespace, pretty=False)
    properties_path = os.path.join(
        CONFIG_STORAGE_PATH, "cerebro-properties.json")
    properties = json.loads(properties_path)
    num_workers = properties["num_of_workers"]
    return num_workers


def update_model_worker_states(model_id, worker_id, is_model_running=False):
    model_on_worker_id = model_id if is_model_running else -1

    model_states = get_model_states()
    worker_states = get_worker_states()
    model_on_worker = get_model_on_worker()

    model_states[model_id] = is_model_running
    worker_states[worker_id] = is_model_running
    model_on_worker[worker_id] = model_on_worker_id

    set_model_states(model_states)
    set_worker_states(worker_states)
    set_model_on_worker(model_on_worker)

    logging.info("-------- Updated model and worker states -------- ")
    logging.info("model_states[{}]: {}".format(
        model_id, model_states[model_id]))
    logging.info("worker_states[{}]: {}".format(
        worker_id, worker_states[worker_id]))
    logging.info("model_on_worker[{}]: {}".format(
        worker_id, model_on_worker[worker_id]))
    logging.info("------------------------------------------------- ")


def scheduler():
    model_worker_pairs = get_model_worker_pairs()
    num_workers = get_num_workers()

    while len(model_worker_pairs) > 0:
        logging.info("\n*************** (Model, Worker) Queue: {}".format(
                     model_worker_pairs))
        reqs = []
        for worker_id in range(num_workers):
            worker_states = get_worker_states()
            if not worker_states[worker_id]:
                # When worker is idle
                runnable_model_id = get_runnable_model(worker_id)
                logging.info("runnable_model_id, worker_id: {} {}".format(
                             runnable_model_id, worker_id))
                if runnable_model_id != -1:
                    update_model_worker_states(
                        runnable_model_id, worker_id, True)
                    data = {
                        "model_id": runnable_model_id,
                        "epoch_number": get_epoch_number()
                    }
                    r = grequests.post(WORKER_ENDPOINT.format(worker_id) +
                                       "/launch-sub-epoch", json=data)
                    reqs.append(r)
            else:
                # Remove (model, worker) pair when the task is complete
                model_on_worker = get_model_on_worker()
                running_model_id = model_on_worker[worker_id]
                logging.info("running_model_id, worker_id: {} {}".format(
                             running_model_id, worker_id))

                params = {'model_id': str(running_model_id)}
                data = requests.get(WORKER_ENDPOINT.format(worker_id) + "/model-status",
                                    params=params)
                status = data.json()['status']

                # Update the model_worker_pairs queue when a task is complete
                logging.info("Task status = {}".format(status))
                if status == 'COMPLETE':
                    model_worker_pairs = get_model_worker_pairs()
                    model_worker_pairs.remove([running_model_id, worker_id])
                    set_model_worker_pairs(model_worker_pairs)
                    update_model_worker_states(
                        running_model_id, worker_id, False)

        rs = grequests.map(reqs, size=num_workers)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
