import tvm
from tvm.topi import tag
from Heron.schedule.sched_op import get_op_methods
from Heron.schedule.primitives import *
from Heron.utils import mapNametoStage, mapNametoAxis, genKey

class Context:
    def __init__(self, knob_manager, build_kwargs, target_name):
        self.sched_desc = ""
        self.codegen_type = None
        self.target_name = target_name
        self.scheduled_axes = []
        self.build_kwargs = build_kwargs
        self.pos_via_tag = {}
        self.tensor_dict = {}
        self.input_tensors = []
        self.axis_anotations = {}
        self.stage_orgnize = None
        self.no_schedule_stages = []

        self.inlined_stages = []
        self.vectorized_stages = []
        self.unrolled_stages = []
        self.general_tile_stages = []
        self.tensorized_stages = []
        self.tiled_stages = []
        self.stile_structures = {}
        self.rtile_structures = {}
        self.unroll_pragma_desc = {}

        # Cache positions
        self.compute_poses = {}
        self.compute_pos_names = {}

        self.tensorize_info = None
        self.knob_manager = knob_manager

    def init_tensor_dict(self, outs):
        if len(outs) == 0:
            return
        inputs = []
        for tensor in outs:
            self.tensor_dict[tensor.name] = tensor
            input_tensors = tensor.op.input_tensors
            inputs += input_tensors
            if len(input_tensors) == 0 and tensor not in self.input_tensors:
                self.input_tensors.append(tensor)
            self.init_tensor_dict(inputs)

    def addSTileStucture(self, stage_name, s_keys, tag):
        assert tag in all_tags
        if stage_name not in self.stile_structures.keys():
            self.stile_structures[stage_name] = []
        self.stile_structures[stage_name].append((s_keys, tag))

    def addRTileStucture(self, stage_name, r_keys, tag):
        assert tag in all_tags
        if stage_name not in self.rtile_structures.keys():
            self.rtile_structures[stage_name] = []
        self.rtile_structures[stage_name].append((r_keys, tag))

    def addSchedDesc(self, strs, add2Solver = False):
        self.sched_desc += strs

    def pos_process(self, s):
        pass

    def addSched(self, op_name, stage_name, s):
        self.knob_manager.sched_tups.append((op_name, stage_name))
        op_methods = get_op_methods(self.codegen_type)
        action = op_methods[op_name](op_name)
        action.perform(s, stage_name, self)

    def updateAxisLength(self, sch):
        for stage in sch.stages:
            axes = stage.leaf_iter_vars
            if stage.op.name in self.compute_poses.keys():
                stage_fused = True
            else:
                stage_fused = False
            for ax in axes:
                if ax.dom == None:
                    continue
                dom_min, dom_extent = ax.dom.min, ax.dom.extent
                assert isinstance(dom_min, tvm.tir.expr.IntImm)
                assert isinstance(dom_extent, tvm.tir.expr.IntImm)
                key = genKey("L", stage.op.name, ax.var.name)
                self.knob_manager.axis_ori_lenth[key] = int(dom_extent) - int(dom_min)
                if stage_fused and ax.iter_type == ax.DataPar:
                    self.knob_manager.staged_fused_axes.add(key)
