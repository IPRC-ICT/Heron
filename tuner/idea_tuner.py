import os
import math
import time
from .tuner import Tuner
from Heron.sample import *
from Heron.utils import *
import random
import numpy as np
from Heron.multi import Job

class IDEATuner(Tuner):
    def optimize(self, env, pop, stat, s_time):
        alpha = 0.2
        N = len(pop)
        N_inf = int(alpha * N)
        N_f = N - N_inf

        all_pop = [] + pop
        for i in range(self.config.iter_walks):
            if self.cost_model.bst == None:
                break
            start_time = time.time()

            # IDEA main body.
            child_pop = self.crossoverAndMutate(pop, env.task)
            pop_f, pop_inf = self.split(pop + child_pop)
            pop_f = self.Rank(pop_f)
            pop_inf = self.Rank(pop_inf)
            pop = pop_f[:N_f] + pop_inf[:N_inf]

            all_pop += child_pop
            aperfs = np.array([x.predict for x in all_pop])
            tup = [aperfs.max(), time.time() - s_time]
            stat.append(tup)
            print(" GA ITER time ", time.time() - start_time)   
        self.removeInvalid(all_pop)
        return pop, all_pop

    def UpdatePopulation(self, env, population):
        # Best K programs in history.
        k = self.config.history_topk
        topk = self.history_topk_samples(env, k)
        # Constrained Random sampling.
        rand = self.constrained_random_sample(env, self.config.pop_num)
        # Update predicted fitness value since cost model has been changed.
        self.repredict(population)

        pop = population + topk + rand
        ret1 = sorted(population + topk, key=lambda x : -x.predict)[:self.config.pop_num//2]
        ret2 = sorted(rand, key=lambda x : -x.predict)[:self.config.pop_num//2]
        pop = ret1 + ret2
        perfs = np.array([x.predict for x in pop])
        print('PMAX %f, PMin %f, NUM %d'%(perfs.max(), perfs.min(), len(perfs)))
        return pop

    def split(self, pop):
        pop_f = []
        pop_inf = []
        for p in pop:
            if p.valid:
                pop_f.append(p)
            else:
                pop_inf.append(p)
        return pop_f, pop_inf

    def constraint_measure(self, pop):
        dim = len(pop[0].knob_manager.solver.primitives)
        for p in pop:
            p.ranks = [0]*dim
            if p.valid and p.violations == None:
                p.violations = [0]*dim
            elif not p.valid and p.violations == None:
                p.violations = [1e8]*dim
        for i in range(dim):
            vs = [x.violations[i] for x in pop]
            sort_idxs = np.argsort(vs)
            rank = -1; prev_score = -1
            for j, sample_id in enumerate(sort_idxs):
                score = vs[sample_id]
                if score > prev_score:
                    rank = rank + 1
                    prev_score = score
                pop[sample_id].ranks[i] = rank
        for p in pop:
            p.rank = sum(p.ranks)

    def undominated_rank_init(self, pop):
        self.constraint_measure(pop)
        l = len(pop)
        for p in pop:
            p.Sp = set()
            p.np = 0
        for i, pi in enumerate(pop):
            for j in range(i+1, l):
                pj = pop[j]
                dominate = pi.predict >= pj.predict and pi.rank >= pj.rank
                rdominate = pj.predict >= pi.predict and pj.rank >= pi.rank
                equal = pi.predict == pj.predict and pi.rank == pj.rank
                if dominate and not equal:
                    pi.Sp.add(pj)
                    pj.np += 1
                elif rdominate and not equal:
                    pi.np += 1
                    pj.Sp.add(pi)

    def crowdingDist(self, pop):
        if len(pop) == 0:
            return
        inf = 1e8
        Ns = len(pop)
        for p in pop:
            p.dist = 0
        pop = sorted(pop, key = lambda x : x.predict)
        fmax = pop[-1].predict
        fmin = pop[0].predict
        assert fmax >= fmin
        pop[0].dist = inf
        pop[-1].dist = inf
        for i in range(1, Ns - 1):
            pop[i].dist = pop[i].dist + (pop[i+1].predict - pop[i-1].predict) / (fmax - fmin + 1e-8)
        # Constraints
        pop = sorted(pop, key = lambda x : x.rank)
        fmax = pop[-1].rank
        fmin = pop[0].rank
        assert fmax >= fmin
        for i in range(1, Ns - 1):
            pop[i].dist = pop[i].dist + (pop[i+1].rank - pop[i-1].rank) / (fmax - fmin + 1e-8)


    def Rank(self, pop):
        self.undominated_rank_init(pop)
        F = [x for x in pop if x.np == 0]
        all_classified = False
        pop_sorted = []
        while not all_classified:
            self.crowdingDist(F)
            NF = []
            for p in F:
                for l in p.Sp:
                    l.np -= 1
                    if l.np == 0:
                        NF.append(l)
            all_classified = len(NF) == 0
            pop_sorted += sorted(F, key = lambda x : x.dist)
            F = NF
        return pop_sorted

    def removeInvalid(self, all_pop):
        for sample in all_pop:
            if not sample.valid:
                sample.predict = 0.0
    
    def crossoverAndMutate(self, samples, task):
        if self.config.parallel:
            res = crossoverAndMutateParallel('SBX', self.config.pop_num,\
                                                 samples, task, self.config.parallel_num, self.feasible)
        else:
            res = crossoverAndMutateSequential('SBX', self.config.pop_num,\
                                                 samples, task, self.feasible)
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
        if kind == 'SBX':
            point, valid, violations = new_sample.knob_manager.SBX_crossover_and_mutation(m1, m2, args)
        new_sample.point = point
        new_sample.violations = violations
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




