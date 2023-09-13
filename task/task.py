try:
    import vta
except:
    import tvm as vta
import tvm
import time
import json
from Heron.utils import *
from Heron.schedule.sched_op import get_op_methods, sched_via_rule
from Heron.schedule.constraints import KnobManager
from Heron.schedule.context import buildContext

class Task:
    def __init__(self, name, opfunc, args, target, target_host):
        self.name = name
        self.args = args
        self.opfunc = opfunc
        self.target = target
        self.target_host = target_host
        self.build_kwargs = None
        self.knob_manager =KnobManager()
        self.not_verify_correctness = True
        self.ref_input = None
        self.ref_output = None
        self.config = None

    def code(self):
        return self.name

    def apply_best(self, path):
        to_sort = []
        for row in open(path):
            json_dict = json.loads(row)
            perf = json_dict['perf']
            param = json_dict['param']
            to_sort.append((perf, param))
        sorted_pairs = sorted(to_sort, key=lambda x : -x[0])
        self.knob_manager.solved_knob_vals_phenotype = sorted_pairs[0][1]
        self.knob_manager.runtime_perf = sorted_pairs[0][0]
        self.knob_manager.is_building = False
        return self.instantiate(self.knob_manager)

    def apply_sample(self, sample):
        self.knob_manager.solved_knob_vals_phenotype = sorted_pairs[0][1]
        self.knob_manager.runtime_perf = sorted_pairs[0][0]
        self.knob_manager.is_building = False
        return self.instantiate(self.knob_manager)       

    def getOut(self):
        ctx = buildContext(self.config.codegen_type, self.knob_manager, self.build_kwargs, str(self.target))
        outs = self.opfunc(ctx, *self.args)
        return outs
    
    def make_stage_schedules(self):
        ctx = buildContext(self.config.codegen_type, self.knob_manager, self.build_kwargs, str(self.target))
        print(self.args)
        outs = self.opfunc(ctx, *self.args)
        s = tvm.te.create_schedule([x.op for x in outs])
        ctx.init_tensor_dict(outs)
        sched_via_rule(self.config.codegen_type, ctx, s)
      # print(tvm.lower(s, ctx.input_tensors + outs))
        self.sched_desc = ctx.sched_desc
        self.knob_manager.is_building = False
        valid, _ = self.knob_manager.solver.solve({}, {})
        self.dump_constraints()
        assert valid

    def dump_constraints(self):
        path = os.path.join(self.config.log_dir, 'constraints.py')
        with open(path, 'w') as f:
            strs = self.knob_manager.solver.dump()
            f.write(strs)

    def instantiate(self, knob_manager):
        ctx = buildContext(self.config.codegen_type, knob_manager, self.build_kwargs, str(self.target))
        outs = self.opfunc(ctx, *self.args)
        s = tvm.te.create_schedule([x.op for x in outs])
        ctx.init_tensor_dict(outs)
        args = ctx.input_tensors + outs
        for tup in knob_manager.sched_tups:
            act_name, stage_name = tup
            op_methods = get_op_methods(self.config.codegen_type)
            action = op_methods[act_name](act_name)
            action.perform(s, stage_name, ctx)
        ctx.pos_process(s)
        knob_manager.sched_desc = ctx.sched_desc
        return s, args

    def set_verify_func(self, func, rtol = 0.1):
        self.rtol = rtol
        self.not_verify_correctness = False
        print("Caculating on cpu to verify correctness")
        self.ref_input, self.ref_output = func(self.args)
        print("End")

    def getStmt(self, knob_manager):
        s, args = self.instantiate(knob_manager)
        if 'vta' in str(self.target):
            return vta.lower(s, args, simple_mode= True)
        else:
            return tvm.lower(s, args, simple_mode= True)


    
