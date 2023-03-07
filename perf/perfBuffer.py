import os
import time
import math
import json


class perfBuffer:
    def __init__(self, config):
        self.perfs = {}
        self.data_x = []
        self.data_y = []
        self.measured_keys = []
        self.best_config = None
        self.best_perf = None
        self.config = config

    def record(self, sample):
        perf = sample.perf
        code = sample.stmt_code
        if code in self.measured_keys:
            return
        self.measured_keys.append(code)
        self.perfs[code] = perf
        if perf != 0:
            self.data_x.append(sample.point)
            self.data_y.append(perf)

        if self.best_perf == None or perf > self.best_perf:
            self.best_perf = perf

        with open(os.path.join(self.config.log_dir, 'records.txt'), 'a') as f:
            f.write(self.encode(sample) + '\n')

    def encode(self, sample):
        json_dict = {
                "code": sample.stmt_code,
                "perf": sample.perf,
                "latency": math.exp(-sample.perf),
                "time": time.time(),
                "param": sample.knob_manager.solved_knob_vals_phenotype,
                } 

        return json.dumps(json_dict)

    def getPerf(self, task):
        code = task.stmt_code
        if code not in self.measured_keys:
            return None
        else:
            if code not in self.perfs.keys():
                return 0.0
            return self.perfs[code]

