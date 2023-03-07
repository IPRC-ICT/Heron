import tvm
from tvm.topi import tag
from .context import Context
from Heron.schedule.sched_op import get_op_methods
from Heron.schedule.primitives import *
from Heron.utils import mapNametoStage, mapNametoAxis

class CPUContext(Context):
    def __init__(self, knob_manager, build_kwargs, target_name):
        Context.__init__(self, knob_manager, build_kwargs, target_name)
        self.parallel_stages = []
        self.cached_stages = []
        self.unpack_info = None
        self.codegen_type = 'CPU' 

    def set_info(self, info):
        if info == None:
            return
        assert info['name'] == 'x86_dot'
        self.tensorize_info = info
        self.stage_orgnize = "X86Tensorize_format"

    def pos_process(self, s):
        for stage_name in self.unroll_pragma_desc.keys():
            ax_name, key = self.unroll_pragma_desc[stage_name]
            stage = mapNametoStage(s, stage_name)
            ax = mapNametoAxis(stage, ax_name)
            unroll_num_exp = self.knob_manager.get_val(key)
            unroll_num = max(1, 4*unroll_num_exp)
            unrollPragma(self, stage, ax, unroll_num, True)
            

    def set_axis_map(self, axis_map):
        self.axis_map = axis_map
