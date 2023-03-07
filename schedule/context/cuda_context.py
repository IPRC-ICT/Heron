import tvm
from tvm.topi import tag
from .context import Context
from Heron.schedule.sched_op import get_op_methods
from Heron.schedule.primitives import *
from Heron.utils import mapNametoStage, mapNametoAxis

class CUDACOREContext(Context):
    def __init__(self, knob_manager, build_kwargs, target_name):
        Context.__init__(self, knob_manager, build_kwargs, target_name)
        self.bind_block_stages = []
        self.bind_thread_stages = []
        self.bind_vthread_stages = []
        self.shared_load_stages = []
        self.default_sharedload_stages = []
        self.codegen_type = 'GPU_CUDA_CORE'
        self.is_tensorcore = False
        self.axis_map = {}
        self.align_sizes = {}
        self.stage_warp_nums = {}

    def set_info(self, info):
        return

    def addThreadLimit(self, s):
        threadx = 'threadIdx.x'
        thready = 'threadIdx.y'
        prod_key = 'threads'
        self.knob_manager.define_value(prod_key, 1, 1024, 1)
        if threadx in self.knob_manager.solver.vals and \
            thready in self.knob_manager.solver.vals:
                self.knob_manager.addProd([threadx, thready], prod_key)

    def addShareMemLimit(self, s):
        for stage_name in self.shared_load_stages:
           #if stage_name not in self.axis_map.keys():
           #    continue
            used_size_name = self.infer_shared_mem_size(s, stage_name)
            if used_size_name not in self.knob_manager.mems:
                self.knob_manager.mems.append(used_size_name)

    def set_axis_map(self, axis_map):
        self.axis_map = axis_map

    def infer_shared_mem_size(self, s, stage_name):
        assert self.is_tensorcore
        assert 'shared' in stage_name
        assert stage_name in self.axis_map.keys()
        gpu_kwargs = self.build_kwargs['check_gpu']
        # For float16 only 
        mem_size = int(gpu_kwargs['max_shared_memory_per_block'] / 2)
        prod_name = stage_name + '_shared_mem_size'
        self.knob_manager.define_value(prod_name, 1, mem_size, 1)
        to_prod = []
        stage = mapNametoStage(s, stage_name)
        for idx, ax in enumerate(stage.op.axis):
            if idx == len(stage.op.axis) - 1 and stage_name in self.align_sizes:
                to_prod.append(self.align_sizes[stage_name])
                continue
            key = stage_name + '_' + ax.var.name
            to_prod.append(key)
        self.knob_manager.addProd(to_prod, prod_name)
        return prod_name

    def InlineAll(self, s):
        visited = set()
        for stage in s.stages:
            op = stage.op
            visited.add(op)
            if tag.is_injective(op.tag):
                self.addSched('compute_inline', stage.op.name, s)

    def getThreadLimit(self, thread_type):
        gpu_kwargs = self.build_kwargs['check_gpu']
        if thread_type == 'threadIdx.x':
            return gpu_kwargs['max_thread_x']
        if thread_type == 'threadIdx.y':
            return gpu_kwargs['max_thread_y']
        if thread_type == 'vthread':
            return 64
        else:
            return 8888888

    def pos_process(self, s):
        for stage_name in self.unroll_pragma_desc.keys():
            ax_name, key = self.unroll_pragma_desc[stage_name]
            stage = mapNametoStage(s, stage_name)
            ax = mapNametoAxis(stage, ax_name)
            unroll_num_exp = self.knob_manager.get_val(key)
            unroll_num = 4**unroll_num_exp
            unrollPragma(self, stage, ax, unroll_num, False)

class TENSORCOREContext(CUDACOREContext):
    def __init__(self, knob_manager, build_kwargs, target_name):
        CUDACOREContext.__init__(self, knob_manager, build_kwargs, target_name)
        self.bind_warp_stages = []
        self.tensorize_loadA_stage = None
        self.tensorize_loadB_stage = None
        self.tensorize_com_stage = None
        self.tensorize_store_stage = None
        self.is_tensorcore = True
        self.codegen_type = 'GPU_TENSOR_CORE'

    def set_info(self, info):
        assert info['name'] == 'tensorcore'
        self.tensorize_info = info
        self.stage_orgnize = "Tensorcore_format"

