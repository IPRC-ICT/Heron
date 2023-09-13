import tvm
from tvm.topi import tag
from Heron.utils import *

def isRootStage(s, stage_name, ctx):
    # If stage is attached by compute_at 
    if stage_name in ctx.compute_poses.keys():
        return False
    # If is a compute node that has not been inlined
    stage = mapNametoStage(s, stage_name)
    out_names = [op.name for op in s.outputs]
    if stage_name not in out_names and stage_name in ctx.inlined_stages:
        return False
    # If stage has no axis return false
    if not hasattr(stage.op, 'axis') or (len(stage.op.axis) == 0 and len(stage.op.reduce_axis) == 0):
        return False
    return True


# Analysis used in Ansor
def hasDataReuse(s, stage_name):
    stage = mapNametoStage(s, stage_name)
    input_tensors = stage.op.input_tensors
    if not isinstance(stage.op, tvm.te.ComputeOp):
        return False
    if len(input_tensors) <= 1:
        return False
    op_loop_num = 0
    if hasattr(stage.op, 'axis'):
        op_loop_num += len(stage.op.axis)
    if hasattr(stage.op, 'reduce_axis'):
        op_loop_num += len(stage.op.reduce_axis)
    for tensor in input_tensors:
        if len(tensor.shape) < op_loop_num:
            return True
    return False

def isPerfectlyMatch(stage, p_stage):
    # Producer's axis should match cur stage perfectly
    for idx, ax in enumerate(stage.leaf_iter_vars):
        c_min, c_extent = ax.dom.min, ax.dom.extent
        c_type = ax.iter_type
        p_ax = p_stage.leaf_iter_vars[idx]
        p_min, p_extent = p_ax.dom.min, p_ax.dom.extent
        p_type = p_ax.iter_type
        if c_min != p_min or \
           c_extent != p_extent or \
           c_type != p_type:
            return False
    return True


def getFusibleProducer(s, stage_name, ctx):
    producers = getProducers(s, stage_name)
    if len(producers) != 1:
        return None
    producer = producers[0]
    p_stage = mapNametoStage(s, producer)
    stage = mapNametoStage(s, stage_name)
    # Producer should have more or equal axes than current
    if len(p_stage.leaf_iter_vars) == 0 or \
       len(stage.leaf_iter_vars) == 0 or\
       len(stage.leaf_iter_vars) > len(p_stage.leaf_iter_vars):
        return None
    if isPerfectlyMatch(stage, p_stage):
        return producer
    elif isUnPacked(stage, p_stage, ctx):
        return producer
    else:
        return None
    return producer


def hasMoreReduce(s, stage_name):
    stage = mapNametoStage(s, stage_name)
    if not hasattr(stage.op, 'reduce_axis') or len(stage.op.axis) == 0:
        return False
    if not hasattr(stage.op, 'axis'):
        return True
    spatial_lenth = 1
    for ax in stage.op.axis:
       extent = ax.dom.extent.value
       spatial_lenth *= extent
    if spatial_lenth < 1024:
        return True
    return False
