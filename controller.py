import os
import pickle
import json
import dill
import string
import pprint
import base64
import random
import logging
import itertools
from time import sleep
import xmlrpc.client as xc

logging.basicConfig(filename="controller.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

namespace = 'cerebro'

CHECKPOINT_STORAGE_PATH = "./cerebro_checkpoint_storage/"
CONFIG_STORAGE_PATH = "./properties/"

class CerebroController:
    def __init__(self, worker_ips, num_epochs, param_grid, train_partitions, valid_partitions, model_fn, input_fn):
        self.worker_ips = worker_ips
        self.num_epochs = num_epochs
        self.param_grid = param_grid

        self.train_partitions = train_partitions
        self.valid_partitions = valid_partitions
        self.model_fn = model_fn
        self.input_fn = input_fn

    def get_worker_ips(self):
        return self.worker_ips


    def get_num_epochs(self):
        return self.num_epochs


    def get_param_grid(self):
        return self.param_grid


    def get_model_worker_pairs(self):
        path = os.path.join(CONFIG_STORAGE_PATH, "model-worker-pairs.pkl")
        
        model_worker_pairs = None
        with open(path, 'rb') as fp:
            model_worker_pairs = pickle.load(fp)
        return model_worker_pairs


    def set_model_worker_pairs(self, model_worker_pairs):
        path = os.path.join(CONFIG_STORAGE_PATH, "model-worker-pairs.pkl")
        
        with open(path, 'wb') as fp:
            pickle.dump(model_worker_pairs, fp)


    def get_model_on_worker(self, ):
        path = os.path.join(CONFIG_STORAGE_PATH, "model-on-worker.pkl")
        
        model_on_worker = None
        with open(path, 'rb') as fp:
            model_on_worker = pickle.load(fp)
        return model_on_worker


    def set_model_on_worker(self, model_on_worker):
        path = os.path.join(CONFIG_STORAGE_PATH, "model-on-worker.pkl")

        with open(path, 'wb') as fp:
            pickle.dump(model_on_worker, fp)


    def get_worker_on_model(self):
        path = os.path.join(CONFIG_STORAGE_PATH, "worker_on_model.pkl")
        
        worker_on_model = None
        with open(path, 'rb') as fp:
            worker_on_model = pickle.load(fp)
        return worker_on_model


    def set_worker_on_model(self, worker_on_model):
        path = os.path.join(CONFIG_STORAGE_PATH, "worker_on_model.pkl")
        
        with open(path, 'wb') as fp:
            pickle.dump(worker_on_model, fp)


    def get_config_on_model(self):
        path = os.path.join(CONFIG_STORAGE_PATH, "config_on_model.pkl")
        
        config_on_model = None
        with open(path, 'rb') as fp:
            config_on_model = pickle.load(fp)
        return config_on_model


    def set_config_on_model(self, config_on_model):
        path = os.path.join(CONFIG_STORAGE_PATH, "config_on_model.pkl")
        
        with open(path, 'wb') as fp:
            pickle.dump(config_on_model, fp)
        

    def get_execid_on_worker(self):
        path = os.path.join(CONFIG_STORAGE_PATH, "execid_on_worker.pkl")
        
        execid_on_worker = None
        with open(path, 'rb') as fp:
            execid_on_worker = pickle.load(fp)
        return execid_on_worker


    def set_execid_on_worker(self, execid_on_worker):
        path = os.path.join(CONFIG_STORAGE_PATH, "execid_on_worker.pkl")
        
        with open(path, 'wb') as fp:
            pickle.dump(execid_on_worker, fp)


    def get_model_on_checkpoint(self):
        path = os.path.join(CONFIG_STORAGE_PATH, "model_on_checkpoint.pkl")
        
        model_on_checkpoint = None
        with open(path, 'rb') as fp:
            model_on_checkpoint = pickle.load(fp)
        return model_on_checkpoint


    def set_model_on_checkpoint(self, model_on_checkpoint):
        path = os.path.join(CONFIG_STORAGE_PATH, "model_on_checkpoint.pkl")
        
        with open(path, 'wb') as fp:
            pickle.dump(model_on_checkpoint, fp)



    def uuid(self):
        """
        Utility function to generate unique identifier
        :return:
        """
        return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(32))


    def check_finished(self, worker, exec_id):
        result = worker.status(exec_id)
        status = dill.loads(base64.b64decode(result.data))
        if status["status"] == "COMPLETED":
            return True, status
        else:
            return False, status


    def get_runnable_model(self, w, models, model_on_worker, mw_pair):
        runnable_model = -1
        random.shuffle(models)
        for m in models:
            if not (mw_pair[m][w]):
                if model_on_worker[m] == -1:
                    runnable_model = m
                    break
        return runnable_model


    def preload_data(self, workers, input_fn_string, train_partitions):
        for i, worker in workers.items():
            worker.initialize_worker()

        exec_ids = []
        for worker_id, worker in workers.items():
            exec_id = self.uuid()
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


    def find_combinations(self):
        param_grid = self.get_param_grid()
        param_keys = list(param_grid.keys())

        params_list = [param_grid[key] for key in param_keys]
        combinations = list(itertools.product(*params_list))
        
        param_combinations = []
        for comb in combinations:
            d = {}
            for i in range(len(comb)):
                d[param_keys[i]] = comb[i]
            param_combinations.append(d)

        return param_combinations


    def init_epoch(self):
        worker_ips = self.get_worker_ips()
        nworkers = len(worker_ips)

        initial_msts = self.find_combinations()
        logging.info("Initial msts: " + pprint.pformat(initial_msts))
        print(initial_msts)

        model_id_to_mst_mapping = {}
        current_msts = [(mst_id, mst) for mst_id, mst in enumerate(initial_msts)]
        for (mst_id, mst) in current_msts:
            model_id_to_mst_mapping[mst_id] = mst

        models_list = list(model_id_to_mst_mapping.keys())
        logging.info("Models list: " + str(models_list))
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

        self.set_config_on_model(model_id_to_mst_mapping)
        self.set_model_on_worker(model_on_worker)
        self.set_worker_on_model(worker_on_model)
        self.set_execid_on_worker(exec_id_on_worker)
        self.set_model_worker_pairs(mw_pair)
        self.set_model_on_checkpoint(model_id_ckpt_mapping)


    def launch_job(self, worker, input_data_path, model_checkpoint_path, 
                input_fn_string, model_fn_string, model_config):
        exec_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(32))
        params = [input_data_path, model_checkpoint_path, input_fn_string, model_fn_string, model_config]
        result = worker.train_model_on_worker(exec_id, params)
        return exec_id


    def scheduler(self, workers):
        train_partitions = self.train_partitions
        valid_partitions = self.valid_partitions
        model_fn = self.model_fn
        input_fn = self.input_fn

        model_id_to_mst_mapping = self.get_config_on_model()
        model_on_worker = self.get_model_on_worker()
        worker_on_model = self.get_worker_on_model()
        mw_pair = self.get_model_worker_pairs()
        model_id_ckpt_mapping = self.get_model_on_checkpoint()
        exec_id_on_worker = self.get_execid_on_worker()

        nworkers = len(workers)
        
        models_list = list(model_id_to_mst_mapping.keys())
        model_to_build = set(model_id_to_mst_mapping.keys())

        model_fn_string = base64.b64encode(dill.dumps(model_fn, byref=False)).decode("ascii")
        input_fn_string = base64.b64encode(dill.dumps(input_fn, byref=False)).decode("ascii")
        
        while (len(model_to_build) > 0):
            for w in range(nworkers):
                if (worker_on_model[w] == -1):
                    m = self.get_runnable_model(w, models_list, model_on_worker, mw_pair)
                    if m != -1:
                        exec_id = self.launch_job(workers[w],
                                            train_partitions[w],
                                            model_id_ckpt_mapping[m],
                                            input_fn_string,
                                            model_fn_string,
                                            model_id_to_mst_mapping[m]
                                            )
                        model_on_worker[m] = w
                        worker_on_model[w] = m
                        exec_id_on_worker[w] = exec_id
                        
                        self.set_model_on_worker(model_on_worker)
                        self.set_worker_on_model(worker_on_model)
                        self.set_execid_on_worker(exec_id_on_worker)

                        print("Sent model {} to build on worker {} with config {}".format(str(m), str(w), str(model_id_to_mst_mapping[m])))
                else:
                    # poll since this particular worker is busy
                    m = worker_on_model[w]
                    exec_id = exec_id_on_worker[w]
                    completed, status = self.check_finished(workers[w], exec_id)

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

                    self.set_model_on_worker(model_on_worker)
                    self.set_worker_on_model(model_on_worker)
                    self.set_model_worker_pairs(mw_pair)
                # TODO: write out execution order in standard format: and also replay schedule(to replay any given scheduler)
                sleep(1)


    def grid_search(self):    
        worker_ips = self.get_worker_ips()
        num_epochs = self.get_num_epochs()
        input_fn_string = base64.b64encode(dill.dumps(self.input_fn, byref=False)).decode("ascii")
        workers = {i: xc.ServerProxy(ip) for i, ip in enumerate(worker_ips)}
        self.preload_data(workers, input_fn_string, self.train_partitions)

        for i in range(num_epochs):
            print("EPOCH: " + str(i+1))
            self.init_epoch()
            self.scheduler(workers)
