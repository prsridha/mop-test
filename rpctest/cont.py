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

def preload_data(workers, train_partitions, valid_data_path):
    for i, worker in workers.items():
        worker.initialize_worker()

    exec_ids = []
    for worker_id, worker in workers.items():
        exec_id = uuid()
        print(train_partitions[worker_id])
        params = [train_partitions[worker_id], "train"]
        result = worker.load_worker_data(exec_id, params)
        status = dill.loads(base64.b64decode(result.data))

        exec_ids.append((exec_id, worker_id))
        if status != "LAUNCHED":
            raise Exception("Remote job launch failed. Reason: " + status)
        exec_id = uuid()
        print(valid_data_path)
        params = [valid_data_path, "valid"]
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
    random.shuffle(models)
    for m in models:
        if not (mw_pair[m][w]):
            if model_on_worker[m] == -1:
                runnable_model = m
                break
    return runnable_model

def launch_job(worker, train_data_path, valid_data_path, model_checkpoint_path, 
                train_fn_string, valid_fn_string, model_config, is_last_worker):
    exec_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(32))
    params = [train_data_path, valid_data_path, model_checkpoint_path, train_fn_string, valid_fn_string, model_config, is_last_worker]
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

#TODO: Validation dataset ETL: task parallel validation : what if validation data doesnt fit in memory of worker: bottleneck
def schedule(worker_ips, train_partitions, valid_partitions, 
            train_fn, valid_fn, initial_msts, preload_data_to_mem):
    print(initial_msts)
    workers = {i: xc.ServerProxy(ip) for i, ip in enumerate(worker_ips)}

    current_msts = [(mst_id, mst) for mst_id, mst in enumerate(initial_msts)]
    model_id_to_mst_mapping = {}
    for (mst_id, mst) in current_msts:
         model_id_to_mst_mapping[mst_id] = mst
    nworkers = len(workers)
    nmodels = len(current_msts)
    models_list = list(model_id_to_mst_mapping.keys())
    model_config_num_map = {}
    print(models_list)
    model_on_worker = {}
    for m in models_list:
        model_on_worker[m] = -1
        model_config_num_map[m] = 0
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
    

    train_fn_string = base64.b64encode(dill.dumps(train_fn, byref=False)).decode("ascii")
    valid_fn_string = base64.b64encode(dill.dumps(valid_fn, byref=False)).decode("ascii")
    
    # only one validation dataset
    validation_data_path = valid_partitions[0]

    if preload_data_to_mem:
        preload_data(workers, train_partitions, validation_data_path)

    model_id_ckpt_mapping = {}
    if not os.path.exists("./models/"):
        print("MAKING MODELS DIR")
        os.makedirs("./models/")
    for mst_id, mst in current_msts:
        ckpt_path =  "./models/" + str(mst_id)
        if not os.path.exists(ckpt_path):
            print("MAKING CHECKPOINT DIR")
            os.makedirs(ckpt_path)
        ckpt_path = ckpt_path + "/{}.model".format(mst_id)
        print("Checkpoint Path: " + ckpt_path + "\n")
        model_id_ckpt_mapping[mst_id] = ckpt_path

    while (len(model_to_build) > 0):
        for w in range(nworkers):
            if (worker_on_model[w] == -1):
                m = get_runnable_model(w, models_list, model_on_worker, worker_on_model, mw_pair)
                if m != -1:
                    is_last_worker = False
                    if model_config_num_map[m] == nworkers - 1:
                        is_last_worker = True
                    exec_id = launch_job(workers[w],
                                        train_partitions[w],
                                        validation_data_path,
                                        model_id_ckpt_mapping[m],
                                        train_fn_string,
                                        valid_fn_string,
                                        model_id_to_mst_mapping[m],
                                        is_last_worker
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
                    model_config_num_map[m] += 1
                    mw_pair[m][w] = True
                    model_done = True
                    for i in range(nworkers):
                        if not mw_pair[m][i]:
                            model_done = False
                            break
                    if model_done:
                        model_to_build.remove(m)
            # TODO: write out execution order in standard format: and also replay schedule(to replay any given scheduler)
            sleep(1)
    
    # print("M[0].n", models[0].n)
    # print("M[1].n", models[1].n)


# validation: can do after last sub epoch (track sub epochs): the worker on which the model sees the last shard
# for the last epoch : along with training, do the validation too on the same worker : last sub epoch: longer time than other sub epochs

def find_best_config(nepochs, worker_ips, train_partitions, valid_partitions, 
            train_fn, valid_fn, train_configs, preload_data_to_mem):
        
        for i in range(nepochs):
            print("EPOCH: " + str(i+1))
            schedule(worker_ips, train_partitions, valid_partitions, train_fn, valid_fn, train_configs, preload_data_to_mem=True)
