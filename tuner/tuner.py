import os
import json
import time
import math
import numpy as np
from Heron.utils import *
from Heron.model import XGBoostCostModel 
from Heron.sample import *
from Heron.multi import Job

class Tuner:
    def __init__(self, config):
        self.config = config
        self.iter_no = 0
        self.cost_model = None 
        self.total_sample_time = 0
        self.total_measure_time = 0

    def buildCostModel(self, task):
        self.cost_model = XGBoostCostModel(task, self.config)

    def run(self, env, do_measure=True):
        self.check_feasible_exits(env)
        start_time = time.time()
        total_iters = math.ceil(self.config.max_trials / self.config.measure_time_per_round)
        population = []; stat=[]
        print("= ---- Tuning : Trials %d, Rounds %d ---- ="%(self.config.max_trials, total_iters))
        for iter_no in range(total_iters): 
            print("== Current Round ", iter_no)
            sample_s = time.time()
            initial_population = self.UpdatePopulation(env, population)
            population, all_pop = self.optimize(env, initial_population, stat, start_time)
            sample_e = time.time()

            measure_s = time.time()
            pop = self.epsilon_select(env, all_pop, epsilon=0.6)
            if do_measure and pop != []:
                samples = self.FilterSamples(pop, env)
                self.measure(samples, env)
            measure_e = time.time()

            train_s = time.time()
            if self.cost_model != None and len(env.perf_buffer.data_y) > 0 and do_measure:
                self.cost_model.train(env.perf_buffer)
            train_e = time.time()
            with open(os.path.join(self.config.log_dir, 'stat.txt'), 'a') as f:
                sample_time = sample_e - sample_s
                measure_time = measure_e - measure_s
                train_time = train_e - train_s
                cur_time = time.time() - start_time
                best_perf = env.perf_buffer.best_perf
                if best_perf == None:
                    best_perf = 0
                f.write("%d,%f,%f,%f,%f,%f\n"%(iter_no, sample_time, measure_time,\
                                 train_time, cur_time, best_perf))
        return population, stat

    def optimize(self, env):
        return NotImplementedError()

    def UpdatePopulation(self, env, population):
        # Best K programs in history.
        k = self.config.history_topk
        topk = self.history_topk_samples(env, k)
        # Constrained Random sampling.
        rand = self.constrained_random_sample(env, self.config.pop_num)
        # Update predicted fitness value since cost model has been changed.
        self.repredict(population)

        all_pop = population + topk + rand
        # Select
        pop_sorted = sorted(all_pop, key=lambda x : -x.predict)
        ret = pop_sorted[:self.config.pop_num]
        perfs = np.array([x.predict for x in ret])
        print('PMAX %f, PMin %f, NUM %d'%(perfs.max(), perfs.min(), len(perfs)))
        return ret
    
    def measure(self, samples, env):
        env.evalSamples(samples)

    def constrained_random_sample(self, env, number, timeout=10):
        if self.config.parallel:
            samples = constrained_random_sample_parallel(env.task, number, self.config, timeout)
        else:
            samples = constrained_random_sample_sequential(env.task, number, self.config)
        for sample in samples:
            sample.predict = self.cost_model.predict([sample])[0]
            sample.violation = 0
        return samples


    def check_feasible_exits(self, env, sample_num = 1e3):
        dir_name = self.config.log_dir
        path = self.config.feasible_file_path
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.feasible = json.loads(f.read())
        else:
            # Generate feasible candidates for each knob by random sampling.
            start = time.time()
            print("Feasible not exists, random sample for the first time.")
            population = self.constrained_random_sample(env, sample_num, timeout=200)
            end = time.time()
            print("Time spend ", end - start)
            dict_set = {}
            for sample in population:
                for valname in sample.knob_manager.solved_knob_vals_phenotype.keys():
                    val = sample.knob_manager.solved_knob_vals_phenotype[valname]
                    if valname not in dict_set:
                        dict_set[valname] = set([val])
                    else:
                        dict_set[valname].add(val)
            dict_list = {}
            estimated_size = 1
            for key in dict_set.keys():
                dict_list[key] = sorted(list(dict_set[key]), key=lambda x:x)
                estimated_size *= len(dict_list[key])
            print("Estimated size ", estimated_size)
            with open(path, 'w') as f:
                f.write(json.dumps(dict_list))
            self.feasible = dict_list

    def RouletteWheelSelection(self, samples, num):
        if len(samples) <= num:
            return samples
        predicts = np.array([x.predict for x in samples])
        predicts_ = predicts - predicts.min() + 1e-16
        probs = predicts_ / predicts_.sum()
        selected = np.random.choice(samples, num, p=probs, replace=False)
        return selected.tolist()

    def RankSelection(self, samples, num):
        if len(samples) <= num:
            return samples
        predicts_ = len(samples) - np.arange(len(samples))
        probs = predicts_ / predicts_.sum()
        selected = np.random.choice(samples, num, p=probs, replace=False)
        return selected.tolist()



    def FilterSamples(self, samples, env):
        codes = set()
        ret = []
        for sample in samples:
            code = sample.stmt_code
            if code in codes or code in env.perf_buffer.perfs:
                continue
            codes.add(code)
            ret.append(sample)
        return ret


    def random_walk(self, task, pop, num_per_sample, feasible):
        if self.config.parallel:
            samples = random_walk_parallel('random_walk', task, pop, num_per_sample, [feasible,])
        else:
            samples = random_walk_sequential('random_walk', task, pop, num_per_sample, [feasible,])
        for sample in samples:
            if sample.valid:
                sample.predict = self.cost_model.predict([sample])[0]
            else:
                sample.predict = 0
        return samples
    
    def constrained_random_walk(self, task, pop, num_per_sample, ratio = 0.3):
        if self.config.parallel:
            samples = random_walk_parallel('constrained_random_walk', task, pop, num_per_sample, [ratio,])
        else:
            samples = random_walk_sequential('constrained_random_walk', task, pop, num_per_sample, [ratio,])
        s1 = time.time()
        for sample in samples:
            sample.predict = self.cost_model.predict([sample])[0]
        print("Predict time ", time.time() - s1)
        return samples

    def guided_constrained_random_walk(self, task, pop, num_per_sample, feasible, ratio=1, step_size = 1):
        keys = anaCostModel(self.cost_model, list(task.knob_manager.solver.vals.keys()))
        length = int(ratio * len(keys))
        topk_keys = keys[:length]
        print("==---- TOP%d FEATURES ----=="%length)
        print(topk_keys)

        if self.config.parallel:
            samples = random_walk_parallel('guided_constrained_random_walk', task, pop, num_per_sample, [topk_keys, step_size, feasible])
        else:
            samples = random_walk_sequential('guided_constrained_random_walk', task, pop, num_per_sample, [topk_keys, step_size, feasible])
        for sample in samples:
            sample.predict = self.cost_model.predict([sample])[0]
        return samples

    def history_topk_samples(self, env, k):
        p = [(x, y) for x,y in zip(env.perf_buffer.data_x, env.perf_buffer.data_y)]
        ps = sorted(p, key= lambda x: x[1])[-k:]
        ret = []
        for tup in ps:
            x, y = tup
            code = Code(x) 
            sample = Sample(env.task)
            sample.fromCode(code)
            predict = self.cost_model.predict([sample])[0]
            sample.violation = 0.0
            sample.predict = predict
            sample.point = x
            sample.stmt_code = code
            ret.append(sample)
        return ret

    def repredict(self, pop):
        for sample in pop:
            if sample.valid:
                sample.predict = self.cost_model.predict([sample])[0]
                sample.violation = 0.0
            else:
                sample.predict = 0.0

    def epsilon_select(self, env, all_samples, epsilon): 
        filtered = self.FilterSamples(all_samples, env)
        if len(filtered) == 0:
            return []
        _sorted = sorted(filtered, key = lambda x: -x.predict)
        print("Best predicted ", _sorted[0].predict)
        k = int(self.config.measure_time_per_round * epsilon)
        topk = _sorted[:k]
        left = _sorted[k:]
        ret = topk + self.RouletteWheelSelection(left, self.config.measure_time_per_round - k)
        return ret

def constrained_random_sample_parallel(task, number, config, timeout):
    work_load = int((number + config.parallel_num - 1) // config.parallel_num)
    jobs = []
    for i in range(config.parallel_num):
        jobs.append(Job(constrained_random_sample_sequential, (task, work_load, config), timeout))

    for job in jobs:
        job.start(job.attach_info)

    res = []
    for job in jobs:
        ret = job.get()
        if isinstance(ret, list):
            res += ret
        del job
    return res


def constrained_random_sample_sequential(task, number, config):
    ret = []
    for i in range(number):
        sample = Sample(task)
        valid, point = sample.knob_manager.randSample({})
        sample.point = point
        sample.valid = valid
        code = Code(point)
        sample.stmt_code = code
        ret.append(sample)
    return ret


def random_walk_parallel(kind, task, pop, num_per_sample, args, timeout=100, parallel_num = 4):
    work_load = int((len(pop) + parallel_num - 1) // parallel_num)
    jobs = []
    for i in range(parallel_num):
        start_id = i * work_load
        end_id = (i+1) * work_load
        jobs.append(Job(random_walk_sequential, (kind, task, pop[start_id : end_id], num_per_sample, args), timeout))

    for job in jobs:
        job.start(job.attach_info)

    res = []
    for job in jobs:
        ret = job.get()
        if isinstance(ret, list):
            res += ret
        del job
    return res


def random_walk_sequential(kind, task, pop, num_per_sample, args):
    ret = []
    for sample in pop:
        for i in range(num_per_sample):
            s = time.time()
            new_sample = Sample(task)

            s1 = time.time()
            if kind == 'random_walk':
                point, valid = new_sample.knob_manager.random_walk(sample.knob_manager, *args)
            elif kind == 'constrained_random_walk':
                point, valid = new_sample.knob_manager.constrained_random_walk(sample.knob_manager, *args)
            elif kind == 'guided_constrained_random_walk': 
                point, valid = new_sample.knob_manager.guided_constrained_random_walk(sample.knob_manager, *args)
            else:
                assert 0
#           print("Walk time %f %f"%(time.time() - s1, s1 - s)) 

            new_sample.point = point
            if valid:
                new_sample.valid = True
            else:
                new_sample.valid = False

            code = Code(point)
            new_sample.stmt_code = code

#           print("Sample time ", time.time() - s)
            ret.append(new_sample)
    return ret





