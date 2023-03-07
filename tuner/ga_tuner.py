import os
import time
from .tuner import Tuner
from Heron.sample import *
from Heron.utils import *
import random
import numpy as np
from Heron.multi import Job
import random

class GATuner(Tuner):
    def optimize(self, env, population, stat, s_time):
        all_pop = [] + population
        for i in range(self.config.iter_walks):
            if self.cost_model.bst == None:
                break
            start_time = time.time()
            population = self.RankPop(population)
            population = self.Selection(population, self.config.select_num)
            population = self.crossoverAndMutate(population, env.task)
            all_pop += population
            aperfs = np.array([x.predict for x in all_pop])
            tup = [aperfs.max(), time.time() - s_time]
            stat.append(tup)
            prevs = sorted(all_pop, key = lambda x:x.predict)[-20:]
            population = population + prevs
            print(" GA ITER time ", time.time() - start_time)   
        self.removeInvalid(all_pop)
        return population, all_pop

    def RankPop(self, pop):
        return pop

    def Selection(self, pop, num):
        return self.RouletteWheelSelection(pop, num)

    def sample(self, pop, env):
        all_pop = []
        for i in range(10):
            pop = self.crossoverAndMutate(pop, env.task)
            all_pop += pop
        return all_pop

    def removeInvalid(self, all_pop):
        for sample in all_pop:
            if not sample.valid:
                sample.predict = 0.0
    
    def crossoverAndMutate(self, samples, task):
        if self.config.parallel:
            res = crossoverAndMutateParallel('Unconstrained', self.config.pop_num,\
                                                 samples, task, self.config.parallel_num, self.feasible)
        else:
            res = crossoverAndMutateSequential('Unconstrained', self.config.pop_num,\
                                                 samples, task, self.feasible)
        for sample in res:
            if sample.valid:
                sample.predict = self.cost_model.predict([sample])[0]
            else:
                sample.predict = 0
        return res

class CGATuner(GATuner):
    def check_feasible_exits(self, env, num=1e3):
        return

    def crossoverAndMutate(self, samples, task):
        keys = anaCostModel(self.cost_model, list(task.knob_manager.solver.vals.keys()))
        key_num = int(self.config.crossover_key_ratio * len(keys))
        keys = keys[:key_num]
        if self.config.parallel:
            res = crossoverAndMutateParallel('Constrained', self.config.pop_num, samples, task, self.config.parallel_num, keys)
        else:
            res = crossoverAndMutateSequential('Constrained', self.config.pop_num, samples, task, keys)
        for sample in res:
            sample.predict = self.cost_model.predict([sample])[0]
        return res

class RCGATuner(GATuner):
    def check_feasible_exits(self, env, num=1e3):
        return

    def crossoverAndMutate(self, samples, task):
        keys = list(task.knob_manager.solver.vals.keys())
        random.shuffle(keys)
        key_num = int(self.config.crossover_key_ratio * len(keys))
        keys = keys[:key_num]
        if self.config.parallel:
            res = crossoverAndMutateParallel('Constrained', self.config.pop_num, samples, task, self.config.parallel_num, keys)
        else:
            res = crossoverAndMutateSequential('Constrained', self.config.pop_num, samples, task, keys)
        for sample in res:
            sample.predict = self.cost_model.predict([sample])[0]
        return res


class SRGATuner(GATuner):
    def Selection(self, pop, num):
        return self.RankSelection(pop, num)

    # Stochastic ranking
    def RankPop(self, pop):
        N = 10
        pf = 0.45
        new_pop = [x for x in pop]
        for i in range(N):
            swapped = False
            for j in range(len(pop) - 1):
                u = random.random()
                p1 = new_pop[j]
                p2 = new_pop[j+1]
                assert p1.violation != None
                assert p2.violation != None
                if (p1.violation == 0 and p2.violation == 0) or u < pf:
                    if p1.predict < p2.predict:
                        new_pop[j] = p2
                        new_pop[j+1] = p1
                        swapped = True
                else:
                    if p1.violation > p2.violation:
                        new_pop[j] = p2
                        new_pop[j+1] = p1
                        swapped = True
            if not swapped:
                break
        return new_pop


    def crossoverAndMutate(self, samples, task):
        if self.config.parallel:
            res = crossoverAndMutateParallel('UnconstrainedAll', self.config.pop_num, samples, task, self.config.parallel_num, self.feasible)
        else:
            res = crossoverAndMutateSequential('UnconstrainedAll', self.config.pop_num, samples, task, self.feasible)
        for sample in res:
            sample.predict = self.cost_model.predict([sample])[0]
        return res


# GA Tuner based on SAT-decoding
class SDGATuner(GATuner):
    def crossoverAndMutate(self, samples, task):
        if self.config.parallel:
            res = crossoverAndMutateParallel('SATDecoding', self.config.pop_num, samples, task, self.config.parallel_num, self.feasible)
        else:
            res = crossoverAndMutateSequential('SATDecoding', self.config.pop_num, samples, task, self.feasible)
        for sample in res:
            sample.predict = self.cost_model.predict([sample])[0]
        return res




def crossoverAndMutateParallel(kind, num, samples, task, parallel_num, args):
    work_load = (num + parallel_num - 1) // parallel_num
    jobs = []
    for i in range(parallel_num):
        jobs.append(Job(crossoverAndMutateSequential, (kind, work_load, samples, task, args)))

    for job in jobs:
        job.start(job.attach_info)

    res = []
    for job in jobs:
        ret = job.get()
        if isinstance(ret, list):
            res += ret
        del job
    return res

def crossoverAndMutateSequential(kind, num, samples, task, args):
    res = []
    for i in range(num):
        idx1 = random.randint(0, len(samples) - 1)
        idx2 = random.randint(0, len(samples) - 1)
        m1 = samples[idx1].knob_manager 
        m2 = samples[idx2].knob_manager 
        new_sample = Sample(task)
        violation = None
        if kind == 'Constrained':
            point, valid = new_sample.knob_manager.constrained_crossover_and_mutate(m1, m2, args)
        elif kind == 'Unconstrained':
            point, valid = new_sample.knob_manager.crossover_and_mutate(m1, m2, args)
        elif kind == 'UnconstrainedAll':
            point, valid, violation = new_sample.knob_manager.crossover_and_mutate_all(m1, m2, args)
        elif kind == "SATDecoding":
            point, valid = new_sample.knob_manager.satdecoding_crossover_and_mutate(m1, m2, args)
        new_sample.point = point
        new_sample.violation = violation
        code = Code(point)
        new_sample.stmt_code = code
        if valid:
            new_sample.valid = True
        else:
            new_sample.valid = False
        res.append(new_sample)
       #task.instantiate(new_sample.knob_manager)
       #assert new_sample.knob_manager.valid()
    return res




