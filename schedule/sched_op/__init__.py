from .gpu_tensorcore_operator import all_op_methods as gpu_tensorcore_methods
from .gpu_tensorcore_operator import sched_via_rule as gpu_tensorcore_sched
from .cpu_operator import all_op_methods as cpu_methods
from .cpu_operator import sched_via_rule as cpu_sched

supportted_codegen_types = [
        'GPU_TENSOR_CORE',
        'CPU',
        'VTA'
        ]

def get_op_methods(_type):
    if _type == 'GPU_TENSOR_CORE':
        return gpu_tensorcore_methods
    elif _type == 'VTA':
        from .vta_operator import all_op_methods as vta_methods
        return vta_methods
    elif _type == 'CPU':
        return cpu_methods
    else:
        raise ValueError('Only support ', supportted_codegen_types)

def sched_via_rule(_type, ctx, s):
    if _type == 'GPU_TENSOR_CORE':
        return gpu_tensorcore_sched(ctx, s)
    elif _type == 'VTA':
        from .vta_operator import sched_via_rule as vta_sched
        return vta_sched(ctx, s)
    elif _type == 'CPU':
        return cpu_sched(ctx, s)

