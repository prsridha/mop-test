import logging
from flask import Flask, request

app = Flask(__name__)
logging.basicConfig(filename="worker.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

namespace = 'cerebro'

CHECKPOINT_STORAGE_PATH = "./cerebro_checkpoint_storage/"
CONFIG_STORAGE_PATH = "./cerebro_config_storage/"
HYPERPARAM_STORAGE_PATH = "./hyperparameter_properties/"
WORKER_ENDPOINT = "http://cerebro-service-worker-{}:6000"
OPTIONAL_HYPERPARAMS = ['batch_size']
REQUIRED_HYPERPARAMS = ['learning_rate', 'lambda_value']


@app.route("/health")
def health():
    return "Hi this is worker. I'm working!\n"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9876)
