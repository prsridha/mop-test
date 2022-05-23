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

@celery.task(name="create_task")
def create_task(n):
    print("Got request for n = ", n)
    sleep(n)
    return True

@app.route("/create-task/<n>")
def run_task(n):
    # n = request.args.get('n')
    task = create_task.delay(int(n))
    return jsonify({"task_id": task.id}), 202

@app.route("/tasks/<task_id>", methods=["GET"])
def get_status(task_id):
    print("GOT GET STATUS TASK ID", task_id)
    task_result = celery.AsyncResult(task_id)
    print("NEW_STUFF", task_result)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result
    }
    print("RESULT", json.dumps(result))
    return jsonify(result), 200

@app.route("/send")
def send_req():
    response = requests.get('http://localhost:9998/create-task/2')
    content = json.loads(response.content)
    task_id = content["task_id"]
    print("GOT TASK ID: ", task_id)

    while True:
        response = requests.get("http://localhost:9998/tasks/{}".format(task_id))
        content = json.loads(response.content)
        pprint(content)
        if content["task_status"] == "SUCCESS":
            break
    return "done"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9998)