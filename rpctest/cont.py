import dill
import string
import random
import base64
from time import sleep
import xmlrpc.client as xc

class model:
    def __init__(self, n):
        self.n = n

def adder_one(m):
    sleep(1)
    print("This adder one")
    print("THis is m", m)
    print("THis is type(m)", type(m))

    return m["n"] + 2

def adder_two(m):
    sleep(2)
    print("THis is m", m)
    print("THis is type(m)", type(m))

    print("This adder two")
    return m["n"] + 5

def get_runnable_model(w, models, model_on_worker, worker_on_model, mw_pair):
    runnable_model = -1
    for m in models:
        if not (mw_pair[m][w]):
            if model_on_worker[m] == -1:
                runnable_model = m
                break
    return runnable_model

def launch_job(worker, func, m):
    exec_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(32))
    params = [m]
    result = worker.execute(exec_id, func, params)
    return exec_id

def check_finished(worker, exec_id):
    result = worker.status(exec_id)
    status = dill.loads(base64.b64decode(result.data))
    if status["status"] == "COMPLETED":
        return True, status
    else:
        return False, status

def schedule(worker_ips):
    workers = {i: xc.ServerProxy(ip) for i, ip in enumerate(worker_ips)}
    m1 = model(10)
    m2 = model(20)

    models = [m1, m2]

    adder_one_str = base64.b64encode(dill.dumps(adder_one, byref=False)).decode("ascii")
    adder_two_str = base64.b64encode(dill.dumps(adder_two, byref=False)).decode("ascii")

    worker_func_mapping = [adder_one_str, adder_two_str]

    nworkers = len(workers)
    nmodels = len(models)

    model_on_worker = {0: -1, 1:-1}
    worker_on_model = {0:-1, 1:-1}
    mw_pair = [[False, False], [False, False]]
    model_to_build = set([0, 1])
    models_list = [0, 1]
    exec_id_on_worker = {0: None, 1: None}

    while (len(model_to_build) > 0):
        for w in range(nworkers):
            if (worker_on_model[w] == -1):
                m = get_runnable_model(w, models_list, model_on_worker, worker_on_model, mw_pair)
                if m != -1:
                    exec_id = launch_job(workers[w], worker_func_mapping[w], models[m])
                    model_on_worker[m] = w
                    worker_on_model[w] = m
                    exec_id_on_worker[w] = exec_id
                    print("Sent model {} to build on worker {}".format(str(m), str(w)))
            else:
                # poll since this particular worker is busy
                m = worker_on_model[w]
                exec_id = exec_id_on_worker[w]
                completed, status = check_finished(workers[w], exec_id)
                if completed:
                    models[m].n = status["result"]
                    model_on_worker[m] = -1
                    worker_on_model[w] = -1
                    mw_pair[m][w] = True
                    model_done = True
                    for i in range(nworkers):
                        if not mw_pair[m][i]:
                            model_done = False
                            break
                    if model_done:
                        model_to_build.remove(m)
    
    print("M[0].n", models[0].n)
    print("M[1].n", models[1].n)

def main():
    ip0 = "http://localhost:7777"
    ip1 = "http://localhost:7778"

    schedule([ip0, ip1])

    print("Done")

if __name__ == "__main__":
    main()