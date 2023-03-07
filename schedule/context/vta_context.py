import tvm
from tvm.topi import tag
from .context import Context
from Heron.schedule.sched_op import get_op_methods
from Heron.schedule.primitives import *
from Heron.utils import mapNametoStage, mapNametoAxis

class VTAContext(Context):
    def __init__(self, knob_manager, build_kwargs, target_name):
        Context.__init__(self, knob_manager, build_kwargs, target_name)
        self.codegen_type = 'VTA'

    def set_info(self, info):
        assert info['name'] == 'vta'
        self.tensorize_info = info
