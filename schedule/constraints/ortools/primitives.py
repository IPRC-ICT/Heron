import math
from Heron.utils import removedot
from ortools.sat.python import cp_model

def getVal(val_dict,v):
    if isinstance(v, str):
        assert v in val_dict.keys()
        return val_dict[v]
    elif isinstance(v, int):
        return v
    assert 0

class LE:
    def __init__(self, args, cond = None):
        self.args = args
        self.cond = cond
        assert len(self.args) == 2

    def func(self, solver, model):
        a = solver.getVal(self.args[0])
        b = solver.getVal(self.args[1])
        if self.cond != None:
            cond = solver.getVal(self.cond)
            model.Add(a <= b).OnlyEnforceIf(cond)
            desc = "model.Add(%s <= %s).OnlyEnforceIf(%s)\n"%(removedot(self.args[0]), removedot(self.args[1]), removedot(self.cond))
        else:
            model.Add(a <= b)
            desc = "model.Add(%s <= %s)\n"%(removedot(self.args[0]), removedot(self.args[1]))
        return desc

    def eval(self, val_dict):
        a = getVal(val_dict, self.args[0])
        b = getVal(val_dict, self.args[1])
        if self.cond != None:
            cond = getVal(val_dict, self.cond)
            if cond == 1:
                return max(b - a, 0)
            else:
                return 0
        else:
            return max(b - a, 0)

class LT:
    def __init__(self, args, cond = None):
        self.args = args
        self.cond = cond
        assert len(self.args) == 2

    def func(self, solver, model):
        a = solver.getVal(self.args[0])
        b = solver.getVal(self.args[1])
        if self.cond != None:
            cond = solver.getVal(self.cond)
            model.Add(a < b).OnlyEnforceIf(cond)
            desc = "model.Add(%s < %s).OnlyEnforceIf(%s)\n"%(removedot(self.args[0]), removedot(self.args[1]), removedot(self.cond))
        else:
            model.Add(a < b)
            desc = "model.Add(%s < %s)\n"%(removedot(self.args[0]), removedot(self.args[1]))
        return desc

    def eval(self, val_dict):
        a = getVal(val_dict, self.args[0])
        b = getVal(val_dict, self.args[1])
        if self.cond != None:
            cond = getVal(val_dict, self.cond)
            if cond == 1:
                if b == a:
                    return 1
                else:
                    return max(b - a, 0)
            else:
                return 0
        else:
            return max(b - a, 0)


class EQ:
    def __init__(self, args, cond = None):
        self.args = args
        self.cond = cond
        assert len(self.args) == 2

    def func(self, solver, model):
        a = solver.getVal(self.args[0])
        b = solver.getVal(self.args[1])
        if self.cond != None:
            cond = solver.getVal(self.cond)
            model.Add(a == b).OnlyEnforceIf(cond)
            desc = "model.Add(%s == %s).OnlyEnforceIf(%s)\n"%(removedot(self.args[0]), removedot(self.args[1]), removedot(self.cond))
        else:
            model.Add(a == b)
            desc = "model.Add(%s == %s)\n"%(removedot(self.args[0]), removedot(self.args[1]))
        return desc

    def eval(self, val_dict):
        a = getVal(val_dict, self.args[0])
        b = getVal(val_dict, self.args[1])
        if self.cond != None:
            cond = getVal(val_dict, self.cond)
            if cond == 1:
                return abs(b - a)
            else:
                return 0
        else:
            return abs(b - a)


class NE:
    def __init__(self, args, cond = None):
        self.args = args
        self.cond = cond
        assert len(self.args) == 2

    def func(self, solver, model):
        a = solver.getVal(self.args[0])
        b = solver.getVal(self.args[1])
        if self.cond != None:
            cond = solver.getVal(self.cond)
            model.Add(a != b).OnlyEnforceIf(cond)
            desc = "model.Add(%s != %s).OnlyEnforceIf(%s)\n"%(removedot(self.args[0]), removedot(self.args[1]), removedot(self.cond))
        else:
            model.Add(a != b)
            desc = "model.Add(%s != %s)\n"%(removedot(self.args[0]), removedot(self.args[1]))
        return desc

    def eval(self, val_dict):
        a = getVal(val_dict, self.args[0])
        b = getVal(val_dict, self.args[1])
        if self.cond != None:
            cond = getVal(val_dict, self.cond)
            if cond == 1 and a != b:
                return 0
            else:
                return 1
        else:
            if a != b:
                return 0
            else:
                return 1

class ProdTwo:
    def __init__(self, args):
        self.args = args
        assert len(self.args) == 2
        assert len(self.args[1]) == 2

    def func(self, solver, model, desc=""):
        target_s, prods_s = self.args
        target = solver.getVal(target_s)
        a = solver.getVal(prods_s[0])
        b = solver.getVal(prods_s[1])
        model.AddMultiplicationEquality(target, [a,b])
        desc = "model.AddMultiplicationEquality(%s, %s)\n"%(removedot(target_s), removedot(str(self.args[1]).replace('\'','')))
        return desc

    def eval(self, val_dict):
        target_s, prods_s = self.args
        target = getVal(val_dict, target_s)
        a = getVal(val_dict, prods_s[0])
        b = getVal(val_dict, prods_s[1])
        return abs(target - a*b)

class Sum:
    def __init__(self, args):
        self.args = args
        assert len(self.args) == 2

    def func(self, solver, model):
        target_s, prods_s = self.args
        target = solver.getVal(target_s)
        prods = [solver.getVal(x) for x in prods_s]
        model.Add(sum(prods)==target)
        desc = "model.Add(sum(%s) == %s)\n"%(removedot(str(prods_s).replace('\'', '')), removedot(target_s))
        return desc

    def eval(self, val_dict):
        target_s, prods_s = self.args
        target = getVal(val_dict, target_s)
        s = sum([getVal(val_dict, x) for x in prods_s])
        return abs(target - s)

#class ModEQ:
#    def __init__(self, args):
#        self.args = args
#        assert len(self.args) == 3
#
#    def func(self, solver, model):
#        target = solver.getVal(self.args[0])
#        var = solver.getVal(self.args[1])
#        mod = solver.getVal(self.args[2])
#        model.AddModuloEquality(target, var, mod)
#        desc = "model.AddModuloEquality(%s, %s, %s)"%(removedot(self.agrs[0]), removedot(self.args[1]), removedot(self.args[2]))
#        return desc



            






