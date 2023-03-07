from ortools.sat.python import cp_model
import time
import random
from Heron.utils import removedot

class Variable:
    def __init__(self, name, low, up, item, defv):
        self.low = low
        self.up = up
        self.item = item
        self.defv = defv
        self.name = name

    def __repr__(self):
        return "Var(%s, %d, %d, %d)"%(self.name, self.low, self.up, self.defv)

class Solver:
    def __init__(self):
        self.vals = {}
        self.primitives = []

    def addVal(self, name, low, up, def_val):
        assert name not in self.vals.keys()
        v = Variable(name, low, up, None, def_val)
        self.vals[name] = v

    def getVal(self, v):
        if isinstance(v, str):
            assert v in self.vals.keys()
            return self.vals[v].item
        elif isinstance(v, int):
            return v
        assert 0

    def validate(self, knob_dict):
        # Create model
        model = cp_model.CpModel()

        keys = list(self.vals.keys())
        for key in keys:
            v = self.vals[key]
            if key not in knob_dict:
                v.item = model.NewIntVar(v.low, v.up, key)
            else:
                v.item = model.NewIntVar(knob_dict[v.name], knob_dict[v.name], key)

        primitives = [x for x in self.primitives]
        for idx, p in enumerate(primitives):
            p.func(self, model)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            res = [solver.Value(self.vals[x].item) for x in self.vals.keys()]
            return True, res
        else:
            return False, None

    def SATScore(self, val_dict):
        scores = 0
        for idx, p in enumerate(self.primitives):
            scores += p.eval(val_dict)
        return scores

    def MultiSATScore(self, val_dict):
        valid = True
        scores = []
        for idx, p in enumerate(self.primitives):
            score = p.eval(val_dict)
            if score > 0:
                valid = False
            scores.append(score)
        return scores, valid

    def dump(self):
        strs = "from ortools.sat.python import cp_model\n\n"
        strs += "model = cp_model.CpModel()\n"
        model = cp_model.CpModel()
        for key in self.vals.keys():
            v = self.vals[key]
            v.item = model.NewIntVar(v.low, v.up, key)
            strs += '%s = model.NewIntVar(%d, %d, \'%s\')\n'%(removedot(key), v.low, v.up, removedot(key))
        
        for idx, p in enumerate(self.primitives):
            strs += p.func(self, model)
        strs += "solver = cp_model.CpSolver()\n"
        strs += "status = solver.Solve(model)\n"
        return strs

    def repair(self, val_dict):
        # Create model
        model = cp_model.CpModel()
        # Create Variales
        keys = list(self.vals.keys())
        for key in keys:
            v = self.vals[key]
            v.item = model.NewIntVar(v.low, v.up, key)

        # Add Constraints 
        primitives = [x for x in self.primitives]
        for idx, p in enumerate(primitives):
            p.func(self, model)

        # Add hint part
        for key in val_dict.keys():
            v = self.getVal(key)
            model.AddHint(v, val_dict[key])

        solver = cp_model.CpSolver()
        solver.parameters.cp_model_presolve = False
        solver.parameters.num_search_workers = 1
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            res = [solver.Value(self.vals[x].item) for x in self.vals.keys()]
            return True, res
        else:
            return False, None

    def solve(self, hint, known, bounds = {}):
        s = time.time()
        # Create model
        model = cp_model.CpModel()

        # Create Variales in random order
        keys = list(self.vals.keys())
        random.shuffle(keys)
        for key in keys:
            v = self.vals[key]
            if key in bounds:
                low = max(v.low, bounds[key][0])
                up = min(v.up, bounds[key][1])
            else:
                low = v.low
                up = v.up
            v.item = model.NewIntVar(low, up, key)

        # Add Constraints in random order
        primitives = [x for x in self.primitives]
        random.shuffle(primitives)
        for idx, p in enumerate(primitives):
            p.func(self, model)

        # Add solved part
        keys = list(known.keys())
        random.shuffle(keys)
        for key in keys:
            v = self.getVal(key)
            cands = known[key]
            if len(cands) == 1:
                model.Add(v == cands[0])
            elif len(cands) == 2:
                # cross over
                bool_1 = model.NewIntVar(0, 1, 'bool_1_' + key)
                bool_2 = model.NewIntVar(0, 1, 'bool_2_' + key)
                model.Add(v == known[key][0]).OnlyEnforceIf(bool_1)
                model.Add(v == known[key][1]).OnlyEnforceIf(bool_2)
                model.Add(bool_1 + bool_2 == 1)
            else:
                assert 0

        # Add hint part
        keys = list(hint.keys())
        random.shuffle(keys)
        for key in keys:
            v = self.getVal(key)
            model.AddHint(v, hint[key])

        solver = cp_model.CpSolver()
        solver.parameters.cp_model_presolve = False
        solver.parameters.num_search_workers = 1
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            res = [solver.Value(self.vals[x].item) for x in self.vals.keys()]
            return True, res
        else:
            return False, None



