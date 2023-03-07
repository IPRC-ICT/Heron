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
        pos_key = stage_name + "_elwise_store_pos"
        if pos_key in ctx.knob_manager.knob_names:
            node = ctx.knob_manager.dep_graph.getNode(pos_key)
            attr = node['attr']
            attr.candidates = [1]
            # Fixed store position
            attr.update(True, 1, 1, 1, set(attr.candidates))

        # Load position for data and weight
        pos_key = stage_name + "_acc_pos"
        if pos_key in ctx.knob_manager.knob_names:
            node = ctx.knob_manager.dep_graph.getNode(pos_key)
            attr = node['attr']
            attr.candidates = list(range(len(ctx.compute_pos_names[stage_name])))
            if tensorize_pos_idx != None:
                attr.candidates = [x for x in attr.candidates if x <= tensorize_pos_idx]
            attr.update(True, None, attr.candidates[0], attr.candidates[-1], set(attr.candidates))

class bindCthreadOp(schedOp):
    def claim_knobs(self, ctx, stage_name, keys):
        limit_key = "Cthread_limit"
        ctx.knob_manager.addProdGen(keys, "Cthread")
        ctx.knob_manager.define_const_value(limit_key, 2)
        ctx.knob_manager.addLE('Cthread', limit_key)

    def perform(self, s, stage_name, ctx):
        ctx.addSchedDesc("\n## Bind Cthread\n")
        stage = mapNametoStage(s, stage_name)
        tile_op = TileSpatialOp("tileSpatial")
        tile_op.tag = "Cthread"
        outer, inner, keys = tile_op.perform(s, stage_name, ctx)
        self.claim_knobs(ctx, stage_name, keys)
        # fuse
        fused = fuse(ctx, stage, outer)
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
                key = stage_name + ax.var.name + "const"
                ctx.knob_manager.define_const_value(key, 1)
            elif idx == i:
                key = "align_i"
                ctx.knob_manager.define_const_value(key, align_i)
            elif idx == j:
                key = "align_j"
                ctx.knob_manager.define_const_value(key, align_j)
            keys.append(key)
        s_outer, s_inner = self.tileUnderKeys(stage, spatial_, keys, True, ctx)
        outer_names = [x.var.name for x in s_outer]
        ctx.addSTileStucture(stage_name, outer_names, "tensorize")
        # Tile keys for all reduce axis
        keys = []
        for idx, ax in enumerate(reduce_):
            if idx != k:
                key = stage_name + ax.var.name + "const"
                ctx.knob_manager.define_const_value(key, 1)
            else:
                key = 'align_k'
                ctx.knob_manager.define_const_value(key, align_k)
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

