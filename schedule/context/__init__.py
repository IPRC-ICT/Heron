from .cpu_context import CPUContext
from .vta_context import VTAContext
from .cuda_context import CUDACOREContext, TENSORCOREContext

def buildContext(_type, knob_manager, build_kwargs, target_name):
    if _type == 'GPU_CUDA_CORE':
        return CUDACOREContext(knob_manager, build_kwargs, target_name)
    elif _type == 'GPU_TENSOR_CORE':
        return TENSORCOREContext(knob_manager, build_kwargs, target_name)
    elif _type == 'CPU':
        return CPUContext(knob_manager, build_kwargs, target_name)
    elif _type == 'VTA':
        return VTAContext(knob_manager, build_kwargs, target_name)
    else:
        raise ValueError('Unsupportted code gen type %s'%_type)
