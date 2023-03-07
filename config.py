import os
import json

class Config:
    def __init__(self):
        # Environment configs
        self.target_name = None
        self.verbose = False
        self.measure_time_per_round = 32
        self.use_cost_model = True
        self.out_name = 'out'

        self.parallel = True
        self.parallel_num = 16

        # Common
        self.pop_num = 200
        self.iter_walks = 5
        self.history_topk = 32

        # XGBoost cost model configs
        self.feature_type = 'itervar' 
        self.loss_type = 'reg'

        # Genetic algorithm params
        self.select_num = 32
        self.crossover_key_ratio = 0.14

        # Simulated annealing params
        self.temperature = 0.6

        self.feasible_file_path = 'feasible.json'

    def setEnv(self, env):
        self.num_ops = env.num_ops
        self.log_dir = os.path.join(self.out_name, env.task.name)

def configFromFile(fname):
    with open(fname, "r") as f:
        data = json.load(f)
    config = Config()
    config.out_name = data['config']['out_name']
    config.opt_method = data['config']['method']
    config.max_trials = data['config']['max_trials']
    config.runner_number = data['config']['runner_number']
    config.runner_repeat = data['config']['runner_repeat']
    config.runner_timeout = data['config']['runner_timeout']
    config.build_timeout = data['config']['build_timeout']
    config.in_dtype = data['config']['in_dtype']
    config.out_dtype = data['config']['out_dtype']
    config.cases = data['config']['cases']
    if "temperature" in data["config"].keys():
        config.temperature = data["config"]["temperature"]
    return config

        

