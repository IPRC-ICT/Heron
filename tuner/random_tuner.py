import os
import time
from .tuner import Tuner
import numpy as np

class CRandTuner(Tuner):
    def optimize(self, env, pop, stat, s_time):
        samples = [] + pop
        for i in range(self.config.iter_walks):
            population = self.constrained_random_sample(env, self.config.pop_num)
            samples = samples + population
            self.recordStat(samples, env, s_time, stat)
        population += sorted(samples, key=lambda x : -x.predict)[:20]
        return population, samples

class CRandSampler(Tuner):
    def optimize(self, env, pop, stat, s_time):
        samples = [] + pop
        for i in range(self.config.iter_walks):
            population = self.constrained_random_sample(env, self.config.pop_num)
            samples = samples + population
            for sample in samples:
                sample.predict = 0
            self.recordStat(samples, env, s_time, stat)
        return population, samples



        



