import tvm
import tvm.te as te
from Heron.utils import *
from Heron.schedule.primitives import *
from .sched_common import *
from .ana_common import *
from tvm import topi
from vta.environment import get_env
def find_const_traverse(s, op, ctx, res):
    if topi.tag.is_broadcast(op.tag):
        if not op.same_as(s.outputs[0]):
            if not op.axis:
                res.append(op)
        for tensor in op.input_tensors:
            if isinstance(tensor.op, tvm.te.PlaceholderOp):
                continue
            else:
                find_const_traverse(s, tensor.op, ctx, res)

def find_elwise_traverse(s, op, ctx, res):
    if topi.tag.is_broadcast(op.tag):
        if not op.same_as(s.outputs[0]):
            if op.axis != None:
                res.append(op)
        for tensor in op.input_tensors:
            if isinstance(tensor.op, tvm.te.PlaceholderOp):
                continue
            else:
                find_elwise_traverse(s, tensor.op, ctx, res)

def find_elwise_input_traverse(s, op, ctx, res):
    if topi.tag.is_broadcast(op.tag):
        for tensor in op.input_tensors:
            if isinstance(tensor.op, tvm.te.PlaceholderOp):
                if op not in res:
                    res.append(op)
            else:
                find_elwise_input_traverse(s, tensor.op, ctx, res)

class VTAfinishOp(finishOp):
    def update_compute_at_candidates(self, s, stage_name, ctx):
        if stage_name not in ctx.compute_pos_names.keys():
            return
        tensorize_pos_idx = None
        tag_dict = {}
        if stage_name in ctx.stile_structures.keys(): 
            tups = ctx.stile_structures[stage_name]
            for i, tup in enumerate(tups):
                _, tag = tup
                if tag == "tensorize":
                    tensorize_pos_idx = i
                tag_dict[tag] = i
        # Store position for elwise ops
        pos_key = genKey("P", stage_name, param_name = "elwise_store_pos")
        if pos_key in ctx.knob_manager.knob_names:
            # Fixed store position
            ctx.knob_manager.solver.vals[pos_key].low = 1
            ctx.knob_manager.solver.vals[pos_key].up = 1

        # Load position for data and weight
        pos_key = genKey("P", stage_name, param_name = "acc_pos")
        if pos_key in ctx.knob_manager.knob_names:
            up = ctx.knob_manager.solver.vals[pos_key].up 
            if tensorize_pos_idx != None:
                up = min(tensorize_pos_idx, up)
            ctx.knob_manager.solver.vals[pos_key].up = up

class bindCthreadOp(schedOp):
    def __init__(self, name):
        self.thread_type = "cthread"
        self.name = name

    def perform(self, s, stage_name, ctx):
        ctx.addSchedDesc("\n## Bind Cthread\n")
        stage = mapNametoStage(s, stage_name)
        tile_op = TileSpatialOp("tileSpatial")
        outer, inner, keys = tile_op.perform(s, stage_name, ctx)
        # fuse
        fused = fuse(ctx, stage, outer)
        ax_key = genKey("L", stage_name, str(fused.var.name))
        ctx.knob_manager.solver.vals[ax_key].up = 2
        # bind
        bind(ctx, stage, fused, "cthread")

class mergeElwiseOp(schedOp):
    def perform(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        env = get_env()
        set_scope(ctx, stage, env.acc_scope)
        pragma(ctx, stage, stage.op.axis[0], env.alu)
        self.define_com_pos(stage_name, s[s.outputs[0]], 'elwise_store_pos', ctx)
        

class addCacheReadElwiseOp(schedOp):
    def perform(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        inputs = stage.op.input_tensors
        ctx.addSchedDesc("\n## Cache read ACC\n")
        stage_tensor = ctx.tensor_dict[stage_name]
        env = get_env()
        output = s.outputs[0]
        for inp in inputs:
            if not isinstance(inp.op, tvm.te.PlaceholderOp):
                continue
            cached = cache_read(ctx, inp, env.acc_scope, stage_tensor, s)
            pragma(ctx, s[cached], s[cached].op.axis[0], env.dma_copy)
            self.define_com_pos(cached.name, s[output], 'elwise_store_pos', ctx)

class addMultiScopeSPMOp(schedOp):
    def perform(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        inputs = stage.op.input_tensors
        ctx.addSchedDesc("\n## Multi-Scope SPM cache read\n")
        env = get_env()
        stage_tensor = ctx.tensor_dict[stage_name]
        assert len(inputs) == 2
        data, weight = inputs
        if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
            pad = data
            data = pad.op.input_tensors[0]
        else:
            pad = None
        if pad is not None:
            set_scope(ctx, s[pad], env.inp_scope)
            cdata = pad
        else:
            cached = cache_read(ctx, data, env.inp_scope, stage_tensor, s)
            cdata = cached
        weight_cached = cache_read(ctx, weight, env.wgt_scope, stage_tensor, s)
        pragma(ctx, s[cdata], s[cdata].op.axis[0], env.dma_copy)
        pragma(ctx, s[weight_cached], s[weight_cached].op.axis[0], env.dma_copy)
        self.define_com_pos(weight_cached.name, s[stage_tensor], 'acc_pos', ctx)
        self.define_com_pos(cdata.name, s[stage_tensor], 'acc_pos', ctx)
        # merge stage to output
        output = s.outputs[0]
        self.define_com_pos(stage_name, s[output], 'elwise_store_pos', ctx)
        set_scope(ctx, stage, env.acc_scope)

class makeStorePositionOp(schedOp):
    def perform(self, s, stage_name, ctx):
        env = get_env()
        ctx.addSchedDesc("\n## TILE output to make store position\n")
        stage = mapNametoStage(s, stage_name)
        # Make compute position via tiling
        tiler = TileSpatialOp("tileSpatial")
        outer, inner, keys = tiler.perform(s, stage_name, ctx)
        # set pragma to out
        pragma(ctx, stage, inner[0], env.dma_copy)

class tensorizeOp(schedOp):
    def perform(self, s, stage_name, ctx):
        info = ctx.tensorize_info
        stage = mapNametoStage(s, stage_name)
        intrinsic, align_i, align_j, align_k = info['intrin']
        _, index = info['com']
        i, j, k = index
        spatial_, reduce_ = self.findUnscheduledAxes(ctx, stage_name, stage.leaf_iter_vars)
        # Tile keys for all spatial axis
        keys = []
        for idx, ax in enumerate(spatial_):
            if idx != i and idx != j:
                key = 1
            elif idx == i:
                key = align_i
            elif idx == j:
                key = align_j
            keys.append(key)
        s_outer, s_inner = self.tileUnderKeys(stage, spatial_, keys, True, ctx)
        outer_names = [x.var.name for x in s_outer]
        ctx.addSTileStucture(stage_name, outer_names, "tensorize")
        # Tile keys for all reduce axis
        keys = []
        for idx, ax in enumerate(reduce_):
            if idx != k:
                key = 1
            else:
                key = align_k
            keys.append(key)
        r_outer, r_inner = self.tileUnderKeys(stage, reduce_, keys, True, ctx)
        outer_names = [x.var.name for x in r_outer]
        ctx.addRTileStucture(stage_name, outer_names, "tensorize")

        reorder(ctx, stage, s_outer + r_outer + s_inner + r_inner)
        tensorize_ax = s_inner[i]
        tensorize_vta(ctx, stage, tensorize_ax, intrinsic)

def do_tensorize(s, stage_name, ctx):
    if ctx.tensorize_info == None:
        return False
    info = ctx.tensorize_info
    assert 'vta' in info['name']
    t_stage_name, _ = info['com']
    if stage_name != t_stage_name:
        return False
    return True

def orgnize_stages(s, ctx):
    elwise_inputs = []; elwise = []; const = []
    find_elwise_traverse(s, s.outputs[0], ctx, elwise)
    find_elwise_input_traverse(s, s.outputs[0], ctx, elwise_inputs) 
    find_const_traverse(s, s.outputs[0], ctx, const)
    # cache read multi scope, stage that to be tensorized
    stage_name = ctx.tensorize_info['com'][0]
    ctx.addSched('addMultiScopeSPMOp', stage_name, s)

    # cache read ALU input
    for consumer in elwise_inputs:
        ctx.addSched('addCacheReadElwise', consumer.name, s)

    # Parallel output
    ctx.addSched('bindCthread', s.outputs[0].name, s)

    # Make store position
    ctx.addSched("makeStorePosition", s.outputs[0].name, s)

    # set ewise scope
    for op in elwise:
        ctx.addSched('mergeElwise', op.name, s)

    # Inline const
    for op in const:
        ctx.addSched('compute_inline', op.name, s)

def do_generaltile(s, stage_name, ctx):
    # Should have data reuse
    if not hasDataReuse(s, stage_name):
        return False
    ctx.general_tile_stages.append(stage_name)
    return True

all_op_methods = {
   # Form schedule
   "start" : startOp,
   "finish" : VTAfinishOp,
   # Normal schedule
   "tileSpatial" : TileSpatialOp,
   "generalTile" : generalTileOp,
   "computeAt" : computeAtOp,
   "compute_inline" : InlineOp,
   # VTA 
   "bindCthread" : bindCthreadOp,
   "addCacheReadElwise" : addCacheReadElwiseOp,
   "addMultiScopeSPMOp" : addMultiScopeSPMOp,
   "makeStorePosition" : makeStorePositionOp,
   "mergeElwise" : mergeElwiseOp,
   "tensorize" : tensorizeOp,
   }


def sched_via_rule(ctx, s):
    # First modify stages
    # Currently disable all inline for cpu
    ctx.updateAxisLength(s)
    orgnize_stages(s, ctx)
    ctx.updateAxisLength(s)
    # The schedule each stage in topological order
    stage_ordered = getStageNamesOrdered(s)
    for stage_name in stage_ordered:
        ctx.addSched("start", stage_name, s)
        # Apply schedule to stage according to static analysis   
        if do_tensorize(s, stage_name, ctx):
            ctx.addSched('tensorize', stage_name, s)
        if do_generaltile(s, stage_name, ctx):
            ctx.addSched('generalTile', stage_name, s)
        ctx.addSched("finish", stage_name, s)

