import json
import requests
from time import sleep 
from pprint import pprint
from celery import Celery
from flask import Flask, request, jsonify
from celery.result import AsyncResult

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

@app.route("/health")
def health():
    return "Yes this is celery test working"

@celery.task(name="train_model")
def train_model(model_id, epoch_num):
    print("Got request for model_id = ", model_id, "and epoch_num = ", epoch_num)
    sleep(5)
    data = {
        "model_id": model_id,
        "epoch_num": epoch_num + 1
    }
    return data

@app.route("/run-subepoch", methods=['POST'])
def run_subepoch():
    content = request.json
    model_id = int(content["model_id"])
    epoch_num = int(content["epoch_num"])

    model_building_task = train_model.delay(model_id, epoch_num)

    return jsonify({"task_id": model_building_task.id}), 202

@app.route("/check-status/<task_id>", methods=["GET"])
def check_status(task_id):
    task_result = celery.AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result
    }
    return jsonify(result), 200

@app.route("/send")
def send_req():
    post_data = {
        "model_id": 10,
        "epoch_num": 20
    }
    url = "http://localhost:9998/run-subepoch"
    headers = {'Content-type': 'application/json'}
    response = requests.post(url, data=json.dumps(post_data), headers=headers)
    content = json.loads(response.content)
    task_id = content["task_id"]
    print("GOT TASK ID: ", task_id)

    while True:
        response = requests.get("http://localhost:9998/check-status/{}".format(task_id))
        content = json.loads(response.content)
        pprint(content)
        sleep(1)
        if content["task_status"] == "SUCCESS":
            print(content["task_result"])
            break
    return "done"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9998)