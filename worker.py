import os
import json
import logging
import tensorflow as tf
from celery import Celery
from flask import Flask, request
from celery.result import AsyncResult

dataset = None

def make_celery(app):
    celery = Celery(app.import_name)
    celery.conf.update(app.config["CELERY_CONFIG"])

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

app = Flask(__name__)
app.config.update(CELERY_CONFIG={
    'broker_url': 'redis://localhost:6379',
    'result_backend': 'redis://localhost:6379',
})
celery = make_celery(app)
logging.basicConfig(filename="worker.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

namespace = 'cerebro'

CHECKPOINT_STORAGE_PATH = "./cerebro_checkpoint_storage/"
CONFIG_STORAGE_PATH = "./cerebro_config_storage/"
HYPERPARAM_STORAGE_PATH = "./hyperparameter_properties/"
DATA_STORAGE_PATH = "./cerebro_data_storage/"
# WORKER_ENDPOINT = "http://cerebro-service-worker-{}:6000"
OPTIONAL_HYPERPARAMS = ['batch_size']
REQUIRED_HYPERPARAMS = ['learning_rate', 'lambda_value']


def get_data():
    #TODO: fix this
    path = os.path.join(DATA_STORAGE_PATH, "tr_x.pkl")
    with open(path, 'rb') as f:
        chunk_x = pickle.load(f)
    path = os.path.join(DATA_STORAGE_PATH, "tr_y.pkl")
    with open(path, 'rb') as f:
        chunk_y = pickle.load(f)
             
    return chunk_x, chunk_y

def get_model(model_id):
    path = os.path.join(CHECKPOINT_STORAGE_PATH, str(model_id))
    model = tf.keras.models.load_model(path)
    return model

def get_model_hyperparameters(model_id):
    # config.load_incluster_config()
    # api_instance = client.CoreV1Api()
    # r = api_instance.read_namespaced_config_map(name, namespace, pretty=False)
    f = open(HYPERPARAM_STORAGE_PATH, "r")
    properties = json.loads(f.read())
    f.close()

    hyperparameters = properties[model_id]
    return hyperparameters


@app.route("/update-model-status", methods=['POST'])
def set_model_status(model_id=None, status=None):
    if model_id is None and status is None:
        data = request.json
        model_id = data['model_id']
        status = data['status']
    path = os.path.join(CHECKPOINT_STORAGE_PATH, str(model_id), "running_status.txt")
    with open(path, 'w') as f:
        f.write(str(status))
    return "Success"


@app.route("/health")
def health():
    return "Hi this is worker. I'm working!\n"

@app.route("/model-status")
def get_model_status():
    task_id = request.args.get('task_id')

    task_result = celery.AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result
    }
    return jsonify(result), 200

@celery.task(name="train_model")
def train_model(model, dataset, hyperparams):
    model.fit(dataset['train'], 
             batch_size = hyperparams['batch_size'] if 'batch_size' in hyperparams else None,
             epochs=1,
             validation_data=dataset['val']
             )
    loss, accuracy = model.evaluate(dataset['val'], verbose=2)
    return (model, loss, accuracy)

@app.route("/execute-sub-epoch", methods=['POST'])
def execute_sub_epoch():
    global dataset
    if not dataset:
        dataset = get_data()

    data = request.json
    model = get_model(data["model_id"])
    hyperparams = get_model_hyperparameters(data["model_id"])

    path = os.path.join(CHECKPOINT_STORAGE_PATH, "{}/saved_model.pb".format(data["model_id"]))
    if os.path.exists(path):
        logging.info("Restoring checkpoint for model {} => {}".format(data["model_id"], path))
        # restore_model_state(model, data["model_id"], data["epoch_number"])
        set_model_status(data["model_id"], 'RUNNING')

    # Train the model
    model, loss, accuracy = train_model.delay(model, dataset, hyperparams)

    logging.info("-------------- Executing Sub-Epoch --------------")
    logging.info("Model summary: {}".format(model.summary()))
    logging.info("Loss (validation set): {}".format(loss))
    logging.info("Accuracy (validation set): {}".format(accuracy))
    logging.info("-------------------------------------------------")
    save_model_state(model, data["model_id"], data["epoch_number"])
    return "Success"

@app.route("/update-model-status", methods=['POST'])
def set_model_status(model_id=None, status=None):
    if model_id is None and status is None:
        data = request.json
        model_id = data['model_id']
        status = data['status']
    path = os.path.join(CHECKPOINT_STORAGE_PATH, str(model_id), "running_status.txt")
    with open(path, 'w') as f:
        f.write(str(status))
    return "Success"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9876)
