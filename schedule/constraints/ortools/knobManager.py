import random
import math
import time
from Heron.utils import *
import numpy as np
from .solver import Solver
from .primitives import *
def str_vec(vec):
    strs = "["
    for i in vec:
        strs += str(i) + ","
    strs += "]"
    return strs

class KnobManager:
    def __init__(self):
        self.sched_tups = []
        self.is_building = True
        self.solver = Solver()

        self.axis_parents = {}
        self.axis_brother = {}
        self.axis_ori_lenth = {}
        self.mems = []
        self.staged_fused_axes = set()

        self.knob_names = []
        self.solved_knob_vals_genotype = {}
        self.solved_knob_vals_phenotype = {}
        self.candidates = {}
        self._valid = False

        self.constraint_descs = ""
        self.dump_descs = False

    def updateAxisParents(self, stage_name, ax_name, parents):
        key = stage_name + '_' + ax_name
        p_keys = [stage_name + '_' + parent for parent in parents]
        self.axis_parents[key] = p_keys 

    def get_axis_roots(self, key):
        if key not in self.axis_parents.keys():
            return [key]
        res = []
        for pkey in self.axis_parents[key]:
            res += self.get_axis_roots(pkey)
        return res

    def get_axis_extent(self, s, stage_name, ax_name):
        key = stage_name + "_" + ax_name
        root_keys = self.get_axis_roots(key)
        # Only split non-fused axes
        assert len(root_keys) == 1
        stage = mapNametoStage(s, stage_name)
        root_ax_name = root_keys[0].split(stage_name + '_')[-1]
        root_ax = mapNametoAxis(stage, root_ax_name)
        dom_extent = root_ax.dom.extent
        return dom_extent.value

    def get_ax_key_extent(self, axkey):
        root_keys = self.get_axis_roots(axkey)
        # Only split non-fused axes
        assert len(root_keys) == 1
        root = root_keys[0]
        return self.axis_ori_lenth[root]


    def get_val(self, name):
        if isinstance(name, int):
            return name
        else:
            assert isinstance(name, str)
        if name in self.axis_ori_lenth.keys() and name not in self.staged_fused_axes:
            return self.axis_ori_lenth[name]
        if self.is_building:
            return self.solver.vals[name].defv
        else:
            if name in self.solved_knob_vals_phenotype.keys():
                return self.solved_knob_vals_phenotype[name]
            else:
                return None

    def addSplitFactor(self, ax_key, axo_key, axi_key, knob):
        if not self.is_building:
            return
        self.solver.primitives.append(ProdTwo([self.get_ax(ax_key), [self.get_ax(axo_key), self.get_ax(axi_key)]]))
        self.solver.primitives.append(EQ([self.get_ax(axi_key), knob]))
        self.axis_brother[axi_key] = axo_key
        self.axis_brother[axo_key] = axi_key

        if self.dump_descs:
            strs = "&Prod(%s,%s)"%(ax_key, str_vec([axo_key, axi_key]))
            strs += "&EQ(%s,%s)"%(axi_key, knob)
            self.constraint_descs += strs
    
    def addSplitNparts(self, ax_key, axo_key, axi_key, knob):
        if not self.is_building:
            return
        self.solver.primitives.append(ProdTwo([self.get_ax(ax_key), [self.get_ax(axo_key), self.get_ax(axi_key)]]))
        self.solver.primitives.append(EQ([self.get_ax(axo_key), knob]))
        self.axis_brother[axi_key] = axo_key
        self.axis_brother[axo_key] = axi_key

        if self.dump_descs:
            strs = "&Prod(%s,%s)"%(ax_key, str_vec([axo_key, axi_key]))
            strs += "&EQ(%s,%s)"%(axo_key, knob)
            self.constraint_descs += strs
    
    def addFuse(self, keys, fused_key):
        self.define_value(fused_key, 1, 1000000, 1)
        prod_keys = [self.get_ax(x) for x in keys] 
        self.addProd(prod_keys, fused_key)

    def addProd(self, src, res):
        if not self.is_building:
            return
        if len(src) == 1:
            args = [src[0], res]
            self.solver.primitives.append(EQ(args))
        elif len(src) == 2:
            args = [res, [src[0], src[1]]]
            self.solver.primitives.append(ProdTwo(args))
        else:
            a = src[0]
            if isinstance(res, str):
                up = self.solver.vals[res].up
            else:
                up = res
            for i in range(len(src) - 2):
                temp = str(a) + '_' + str(src[i + 1])
                self.define_value(temp, 1, up, 1)
                args = [temp, [a, src[i + 1]]]
                self.solver.primitives.append(ProdTwo(args))
                a = temp
            args = [res, [temp, src[-1]]]
            self.solver.primitives.append(ProdTwo(args))

        if self.dump_descs:
            strs = "&Prod(%s,%s)"%(res, str_vec(src))
            self.constraint_descs += strs

   #def addDivisable(self, x, mod):
   #    if not self.is_building:
   #        return
   #    self.solver.primitives.append(ModEQ([0, [x, mod]]))

    def addLE(self, a, b):
        if not self.is_building:
            return
        assert isinstance(a, str)
        if isinstance(b, str):
            self.solver.primitives.append(LE([a, b]))
        else:
            self.changeUp(a, b)
        if self.dump_descs:
            strs = "&LE(%s,%s)"%(a, b)
            self.constraint_descs += strs

    def addEQ(self, a, b):
        if not self.is_building:
            return
        assert isinstance(a, str)
        if isinstance(b, str):
            self.solver.primitives.append(EQ([a, b]))
        else:
            self.changeUp(a, b)
            self.changeLow(a, b)
        if self.dump_descs:
            strs = "&EQ(%s,%s)"%(a, b)
            self.constraint_descs += strs

    def addNE(self, a, b):
        if not self.is_building:
            return
        self.solver.primitives.append(NE([a, b]))
        if self.dump_descs:
            strs = "&NE(%s,%s)"%(a, b)
            self.constraint_descs += strs

    def addLT(self, a, b):
        if not self.is_building:
            return
        assert isinstance(a, str)
        if isinstance(b, str):
            self.solver.primitives.append(LT([a, b]))
        else:
            self.changeUp(a, b - 1)

        if self.dump_descs:
            strs = "&LT(%s,%s)"%(a, b)
            self.constraint_descs += strs


    def addSelect(self, cand_keys, knob_key, val_key):
        if not self.is_building:
            return
        up = self.solver.vals[knob_key].up
        up = min(len(cand_keys) - 1, up)
        keys = []
        for i in range(up + 1):
            key = knob_key + '_select%d'%i
            self.define_value(key, 0, 1, 0)
            self.solver.primitives.append(EQ([val_key, cand_keys[i]], cond = key))
            self.solver.primitives.append(EQ([knob_key, i], cond = key))
            keys.append(key)
        self.solver.primitives.append(Sum([1, keys]))

        if self.dump_descs:
            strs = "&SELECT(%s,%s,%s)"%(val_key, knob_key, str_vec(cand_keys))
            self.constraint_descs += strs


    def addSUM(self, resname, keys):
        if not self.is_building:
            return
        self.solver.primitives.append(Sum([resname, keys]))
        if self.dump_descs:
            strs = "&SUM(%s,%s)"%(resname, str_vec(keys))
            self.constraint_descs += strs
            
    def addCandidates(self, valname, candidates):
        if not self.is_building:
            return
        # Build idxs
        idxs = []
        for idx, v in enumerate(candidates):
            idx_name = valname + '_cand%d'%idx
            self.define_value(idx_name, 0, 1, 0)
            self.solver.primitives.append(EQ([valname, candidates[idx]], cond = idx_name))
            idxs.append(idx_name)
        self.solver.primitives.append(Sum([1, idxs]))
        if valname in self.knob_names:
            self.candidates[valname] = candidates
        if self.dump_descs:
            strs = "&IN(%s,%s)"%(valname, str_vec(candidates))
            self.constraint_descs += strs

    # Do not insert restriction
    def addRawCandidates(self, valname, candidates):
        if not self.is_building:
            return
        self.candidates[valname] = candidates 


    def define_value(self, name, low, up, def_val, is_knob = False):
        if not self.is_building:
            return name
        if name in self.solver.vals.keys():
            return name
        if is_knob:
            self.knob_names.append(name)
        self.solver.addVal(name, low, up, def_val)
        return name

    def get_ax(self, ax):
        if ax in self.solver.vals.keys():
            res = ax
        elif ax in self.axis_ori_lenth and not ax in self.staged_fused_axes:
            res = self.axis_ori_lenth[ax]
        else:
            extent = self.get_ax_key_extent(ax)
            res = self.define_value(ax, 1, extent, 1)
        assert res != None
        return res

    def changeUp(self, a, b):
        up = self.solver.vals[a].up
        up = min(up, b)
        self.solver.vals[a].up = up

    def changeLow(self, a, b):
        low = self.solver.vals[a].low
        low = max(low, b)
        self.solver.vals[a].low = low

    def getExtent(self, name):
        assert name in self.solver.vals.keys()
        return self.solver[name]

    def get_knob_candidates(self, name):
        if name in self.candidates.keys():
            return self.candidates[name]
        else:
            low = self.solver.vals[name].low
            up = self.solver.vals[name].up
            assert up - low + 1 <= 30
            return list(range(low, up + 1))

    def randSample(self, known = {}, bounds = {}):
        hint = {}
        for key in self.knob_names:
            if key in known.keys():
                continue
            candidates = self.get_knob_candidates(key)
            idx = random.randint(0, len(candidates) - 1)
            hint[key] = candidates[idx]
        valid, point = self.solver.solve(hint = hint, known = known, bounds = {})
        assert valid
        self._valid = valid
        for idx, key in enumerate(list(self.solver.vals.keys())):
            self.solved_knob_vals_genotype[key] = point[idx] 
            self.solved_knob_vals_phenotype[key] = point[idx] 
        return valid, point

    def validate(self, knob_dict):
        valid, point = self.solver.validate(knob_dict)
        self._valid = valid
        if valid:
            for idx, key in enumerate(list(self.solver.vals.keys())):
                self.solved_knob_vals_genotype[key] = point[idx]
                self.solved_knob_vals_phenotype[key] = point[idx]
        else:
            for key in knob_dict.keys():
                self.solved_knob_vals_genotype[key] = knob_dict[key]
                self.solved_knob_vals_phenotype[key] = knob_dict[key]
            point = [knob_dict[key] for key in knob_dict.keys()]
        return point, valid

    def constrained_crossover_and_mutate(self, m1, m2, keys):
        known = {}
        if len(keys) > 0:
            # mutation idx
            midx = random.randint(0, len(keys) - 1)
            for idx, key in enumerate(keys):
                if idx == midx:
                    continue
                cands = []
                cands.append(m1.solved_knob_vals_genotype[key])
                cands.append(m2.solved_knob_vals_genotype[key])
                known[key] = cands
        valid, res = self.randSample(known) 
        return res, valid

    def onepoint_crossover_and_mutation(self, keys, m1, m2, feasible):
        # Randomly select a point as the crossover position.
        position = random.randint(0, len(keys) - 1)
        m1_keys = keys[:position]
        m2_keys = keys[position:]
        known = {}
        # Randomly select a point to mutate
        midx = random.randint(0, len(keys) - 1)
        mkey = keys[midx]
        res = []
        for key in m1.solved_knob_vals_genotype.keys():
            if key == mkey:
                cands = feasible[key]
                v = cands[random.randint(0, len(cands) - 1)]
                known[key] = v
            elif key in m1_keys:
                known[key] = m1.solved_knob_vals_genotype[key]
            else:
                known[key] = m2.solved_knob_vals_genotype[key]
            res.append(known[key])
        return known, res
   
    def crossover_and_mutate(self, m1, m2, feasible):
        keys = self.knob_names
        known, _ = self.onepoint_crossover_and_mutation(keys, m1, m2, feasible)
        res, valid =  self.validate(known)
        return res, valid
        
    def crossover_and_mutate_all(self, m1, m2, feasible):
        keys = list(m1.solved_knob_vals_genotype.keys())
        known, res = self.onepoint_crossover_and_mutation(keys, m1, m2, feasible)
        self.solved_knob_vals_genotype = known
        self.solved_knob_vals_phenotype = known
        score = self.solver.SATScore(known)
        if score == 0:
            self._valid = True
        else:
            self._valid = False
        return res, self._valid, score
        
    def satdecoding_crossover_and_mutate(self, m1, m2, feasible):
        keys = list(m1.solved_knob_vals_genotype.keys())
        genotype_dict, genotype = self.onepoint_crossover_and_mutation(keys, m1, m2, feasible)
        self.solved_knob_vals_genotype = genotype_dict
        valid, phenotype = self.solver.repair(genotype_dict)
        self._valid = valid
        assert valid
        for idx, key in enumerate(list(self.solver.vals.keys())):
            self.solved_knob_vals_phenotype[key] = phenotype[idx]
        return phenotype, valid

    def SBX_crossover_and_mutation(self, m1, m2, feasible):
        ita1 = 15;ita2 = 20
        keys = [x for x in m1.solved_knob_vals_genotype.keys()]
        # crossover
        res = []; chrom = {}
        for key in keys:
            x1 = m1.solved_knob_vals_genotype[key]
            x2 = m2.solved_knob_vals_genotype[key]
            u = random.random()
            s = float(random.randint(0,1) * 2 - 1)
            if u <= 0.5:
                beta = math.pow(2*u, 1 / ita1 + 1)
            else:
                beta = math.pow(1 / (2*(1 - u + 1e-8)), 1 / ita1 + 1)
            beta *= s
            x = 0.5*((1 + beta)*x1 + (1 - beta)*x2)
            dist = [abs(t - x) for t in feasible[key]]
            idx = np.argmin(dist)
            x = feasible[key][idx]

            # mutate
            if random.random() < 0.1:
                l = np.max(feasible[key]) - np.min(feasible[key])
                r = random.random()
                if r < 0.5:
                    coe = math.pow(2*r, 1 /(ita2 + 1)) - 1
                else:
                    coe = 1 - math.pow(2*(1-r), 1 /(ita2 + 1))
                x = x + l * coe
                dist = [abs(t - x) for t in feasible[key]]
                idx = np.argmin(dist)
                x = feasible[key][idx]
            chrom[key] = x
            res.append(x)
        violations, valid = self.solver.MultiSATScore(chrom)
        self.solved_knob_vals_genotype = chrom
        self.solved_knob_vals_phenotype = chrom
        self._valid = valid
        return res, valid, violations
        

    def constrained_random_walk(self, knob_manager, ratio):
        keys = [x for x in knob_manager.solved_knob_vals_genotype.keys()]
        preserve_num = int(ratio * len(keys))
        # Random select mutation positions
        positions = np.random.choice(len(keys), preserve_num, replace=False).tolist()
        known = {}
        for idx, key in enumerate(keys):
            if idx not in positions:
                continue
            known[key] = [knob_manager.solved_knob_vals_genotype[key]]
        valid, res = self.randSample(known) 
        return res, valid

    def guided_constrained_random_walk(self, knob_manager, topk_keys, step_size, feasible):
        topk_bounds = {}
        for key in topk_keys:
            val = knob_manager.solved_knob_vals_genotype[key]
            cands = feasible[key]
            lt = [x for x in cands if x < val]
            lrange = lt[-step_size:] + [val]
            gt = [x for x in cands if x > val]
            grange = [val] + gt[:step_size]
            topk_bounds[key] = [lrange[0], grange[-1]]
        valid, res = self.randSample({}, topk_bounds) 
        return res, valid

    def random_walk(self, knob_manager, feasible):
        num_knobs = len(self.knob_names)
        knob_name = self.knob_names[random.randint(0, num_knobs - 1)]
        candidates = feasible[knob_name]
        val = candidates[random.randint(0, len(candidates) - 1)]
        point = []
        new_knobs = {}
        for name in self.knob_names:
            if knob_name == name:
                new_knobs[name] = val
            else:
                new_knobs[name] = knob_manager.solved_knob_vals_genotype[name]
            point.append(new_knobs[name])
        res, valid =  self.validate(new_knobs)
        if valid:
            point = res
        return point, valid


    def valid(self):
        return self._valid

         



