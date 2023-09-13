import os
import math
import time
import heapq
from .tuner import Tuner
from Heron.sample import *
from Heron.utils import *
import random
import numpy as np
import json


class SATuner(Tuner):
    def __init__(self, config):
        super().__init__(config)
        self.t = self.config.temperature
        total_iters = math.ceil(self.config.max_trials / self.config.measure_time_per_round) * self.config.iter_walks 
        self.cool = self.t / (total_iters + 1) 

    def optimize(self, env, population, stat, s_time):
        all_samples = [] + population
        for i in range(self.config.iter_walks):
            if hasattr(self.cost_model, "bst") and self.cost_model.bst == None:
                break
            population_1 = self.walk(population, env)
            population = self.accept(population, population_1, self.t)
            all_samples += population

            aperfs = np.array([x.predict for x in all_samples])
            stat.append([aperfs.max(), time.time() - s_time])
            print("Max %f, time %f"%(aperfs.max(), time.time() - s_time))
            self.t -= self.cool
        return population, all_samples

    def sample(self, pop, env):
        all_pop = []
        for i in range(10):
            pop = self.walk(pop, env)
            all_pop += pop
        return all_pop

    def accept(self, pop, pop1, temp):
        score = np.array([x.predict for x in pop])
        score1 = np.array([x.predict for x in pop1])
        ac_prob = np.exp(np.minimum((score1 - score) / ( temp + 1e-5 ), 0))
        ac_index = (np.random.random(len(ac_prob)) < ac_prob).tolist()
        ac_index = (score1 > score).tolist()
    #   print(ac_prob)
        ret = []
        for i in range(len(pop)):
            if ac_index[i]:
                ret.append(pop1[i])
            else:
                ret.append(pop[i])
        return ret


class ConstrainedRandomWalkerSATuner(SATuner):
    def walk(self, pop, env):
        return self.constrained_random_walk(env.task, pop, 1, ratio=0.3)

class RandomWalkSATuner(SATuner):
    def walk(self, pop, env):
        return self.random_walk(env.task, pop, 1, self.feasible)

