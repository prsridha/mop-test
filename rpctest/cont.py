import os
import dill
import string
import random
import base64
from time import sleep
import xmlrpc.client as xc

class model:
    def __init__(self, n):
        self.n = n

dill.settings["recurse"] = True

def preload_data(workers, input_fn_string, train_partitions):
    for i, worker in workers.items():
        worker.initialize_worker()

    exec_ids = []
    for worker_id, worker in workers.items():
        exec_id = uuid()
        print(train_partitions[worker_id])
        params = [input_fn_string, train_partitions[worker_id]]
        result = worker.load_worker_data(exec_id, params)
        status = dill.loads(base64.b64decode(result.data))

        if status != "LAUNCHED":
            raise Exception("Remote job launch failed. Reason: " + status)

        exec_ids.append((exec_id, worker_id))

    # wait for everything to finish
    while len(exec_ids) > 0:
        for exec_id, worker_id in exec_ids:
            worker = workers[worker_id]
            status = dill.loads(base64.b64decode(worker.status(exec_id).data))

            if status["status"] == "FAILED":
                print(status)
                raise Exception("Remote job execution failed")
            elif status["status"] == "INVALID ID":
                raise Exception("Invalid Id")
            elif status["status"] == "COMPLETED":
                exec_ids.remove((exec_id, worker_id))
                message = "EVENT: PRELOAD_COMPLETED, WORKER: %d\n" % (worker_id)
                print(message[:-1])
        sleep(1)



def get_runnable_model(w, models, model_on_worker, worker_on_model, mw_pair):
    runnable_model = -1
    for m in models:
        if not (mw_pair[m][w]):
            if model_on_worker[m] == -1:
                runnable_model = m
                break
    return runnable_model

def launch_job(worker, input_data_path, model_checkpoint_path, 
               input_fn_string, model_fn_string, model_config):
    exec_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(32))
    params = [input_data_path, model_checkpoint_path, input_fn_string, model_fn_string, model_config]
    result = worker.train_model_on_worker(exec_id, params)
    return exec_id

def check_finished(worker, exec_id):
    result = worker.status(exec_id)
    status = dill.loads(base64.b64decode(result.data))
    if status["status"] == "COMPLETED":
        return True, status
    else:
        return False, status

def log_message(log_file, message, print_message=False):
    """
    :param log_file:
    :param message:
    :param print_message:
    """
    log_file.write(message)
    log_file.flush()
    os.fsync(log_file.fileno())
    if print_message:
        print(message[:-1])

def uuid():
    """
    Utility function to generate unique identifier
    :return:
    """
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(32))

def schedule(worker_ips, train_partitions, valid_partitions, 
            input_fn, model_fn, initial_msts, preload_data_to_mem):
    print(initial_msts)
    workers = {i: xc.ServerProxy(ip) for i, ip in enumerate(worker_ips)}

    current_msts = [(mst_id, mst) for mst_id, mst in enumerate(initial_msts)]
    model_id_to_mst_mapping = {}
    for (mst_id, mst) in current_msts:
         model_id_to_mst_mapping[mst_id] = mst
    nworkers = len(workers)
    nmodels = len(current_msts)
    models_list = list(model_id_to_mst_mapping.keys())
    print(models_list)
    model_on_worker = {}
    for m in models_list:
        model_on_worker[m] = -1
    worker_on_model = {}
    exec_id_on_worker = {}
    for w in range(nworkers):
        exec_id_on_worker[w] = None
        worker_on_model[w] = -1

    mw_pair = []
    for m in models_list:
        lis = []
        for w in range(nworkers):
            lis.append(False)
        mw_pair.append(lis)

    
    model_to_build = set(models_list)
    


    input_fn_string = base64.b64encode(dill.dumps(input_fn, byref=False)).decode("ascii")
    model_fn_string = base64.b64encode(dill.dumps(model_fn, byref=False)).decode("ascii")

    if preload_data_to_mem:
        preload_data(workers, input_fn_string, train_partitions)

    model_id_ckpt_mapping = {}
    for mst_id, mst in current_msts:
        ckpt_path =  "./" + str(mst_id) + "_" + uuid()
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        ckpt_path = ckpt_path + "/{}.model".format(mst_id)
        print("Checkpoint Path: " + ckpt_path + "\n")
        model_id_ckpt_mapping[mst_id] = ckpt_path

    while (len(model_to_build) > 0):
        for w in range(nworkers):
            if (worker_on_model[w] == -1):
                m = get_runnable_model(w, models_list, model_on_worker, worker_on_model, mw_pair)
                if m != -1:
                    exec_id = launch_job(workers[w],
                                        train_partitions[w],
                                        model_id_ckpt_mapping[m], 
                                        input_fn_string, 
                                        model_fn_string,
                                        model_id_to_mst_mapping[m]
                                        )
                    model_on_worker[m] = w
                    worker_on_model[w] = m
                    exec_id_on_worker[w] = exec_id
                    print("Sent model {} to build on worker {} with config {}".format(str(m), str(w), str(model_id_to_mst_mapping[m])))
            else:
                # poll since this particular worker is busy
                m = worker_on_model[w]
                exec_id = exec_id_on_worker[w]
                completed, status = check_finished(workers[w], exec_id)
                if completed:
                    print("Received Model {} built on worker {}".format(str(m), str(w)))
                    # models[m].n = status["result"]
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
                
            sleep(5)
    
    # print("M[0].n", models[0].n)
    # print("M[1].n", models[1].n)
