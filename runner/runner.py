import math
import numpy as np
from tvm.autotvm.measure.measure import MeasureInput

class Runner:
    def __init__(self, measure_option):
        self.measure_option = measure_option

    def results2tup(self, results):
        res = []
        for x in results:
          # print("-----------------------------------------------------------------------------")
            if x.error_no == 0:
                perf_time = np.array(x.costs).mean()
                res.append((x.error_no, math.log(1 / perf_time) ))
          #     print(x)
            else:
                res.append((x.error_no, 0.0))
          #     for cost in x.costs:
          #         print(cost)
        return res
    
    def run(self, samples, perf_buffer):
        task = samples[0].task
        for sample in samples:
            assert task.name == sample.task.name
        cfgs = [x.knob_manager for x in samples]
        inps = [MeasureInput(task.target, task, cfg) for cfg in cfgs]
        results = self.measure_batch(inps)
        res = self.results2tup(results)
        for idx, sample in enumerate(samples):
            error, perf = res[idx]
            sample.perf = perf
            if error == 0:
                sample.valid = True
            else:
                sample.valid = False
            perf_buffer.record(sample)

        best = perf_buffer.best_perf
        print('TASK %s, Cur best %f'%(task.name, best))
        for idx, sample in enumerate(samples):
            print("Sample %s"%sample.stmt_code)
            print("Perf %f , Predict %f, Prob %f"%(sample.perf, sample.predict, sample.prob))
