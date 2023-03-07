import copy
import tvm
import time
import hashlib
import numpy as np
from Heron.utils import * 

class Sample:
    def __init__(self, task):
        self.valid = False
        self.perf = 0.0
        self.task = task
        self.knob_manager = copy.deepcopy(task.knob_manager)
        self.predict = 0
        self.prob = 0
        self.violation = None
        self.violations = None
        self.ranks = []

    def getTask(self):
        return self.task

    def fromCode(self, code):
        point = DeCode(code)
        self.point = point
        for idx, key in enumerate(list(self.knob_manager.solver.vals.keys())):
            self.knob_manager.solved_knob_vals_genotype[key] = point[idx]

    def lower(self):
        task = self.getTask()
        sch, fargs = task.instantiate(self.knob_manager)
        stmt = tvm.lower(sch, fargs)
        return stmt



