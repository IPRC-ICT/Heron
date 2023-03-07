import tvm
import tvm.te as te
from Heron.utils import *
from Heron.schedule.primitives import *
from .sched_common import *
from .ana_common import *
class addCacheWriteGlobalOp(schedOp):
    def perform(self, s, stage_name, ctx):
        ctx.addSchedDesc("\n## Cache write global\n")
        stage = mapNametoStage(s, stage_name)
        out = ctx.tensor_dict[stage_name]
        out_chached = cache_write(ctx, out, "global", s)
        self.define_com_pos(out_chached.name, s[out], 'global_pos', ctx)
        ctx.cached_stages.append(stage_name)


class parallelOp(schedOp):
    def perform(self, s, stage_name, ctx):
        # start from outer, stop at reduce
        ctx.addSchedDesc("\n## Parallel \n")
        stage = mapNametoStage(s, stage_name)
        tile_op = TileSpatialOp("tileSpatial")
        outer, inner, keys = tile_op.perform(s, stage_name, ctx)
        fused = fuse(ctx, stage, outer)
        parallel(ctx, stage, fused)
        ctx.parallel_stages.append(stage_name)
        return keys

class tileForCacheOp(schedOp):
    def perform(self, s, stage_name, ctx):
        ctx.addSchedDesc("\n## Tile for cache \n")
        stage = mapNametoStage(s, stage_name)
        tile_op = TileSpatialOp("tileSpatial")
        outer, inner, keys = tile_op.perform(s, stage_name, ctx)
        # Restrict global cache pos
        global_pos_key = stage_name + "_global_pos"
        assert global_pos_key in ctx.knob_manager.knob_names
     #  ctx.knob_manager.addEQ(global_pos_key, 1)
     #  # Caculate cache size, for cost model usage , can exert constraint
     #  key = stage_name + '_cach_size'
     #  ctx.knob_manager.define_value(key, 1, 88888888, 1)
     #  prod_keys = [stage_name + '_' + x.var.name for x in inner]
     #  ctx.knob_manager.addProd(prod_keys, key)


class CPUvectorizeOp(schedOp):
    def perform(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        axes = stage.leaf_iter_vars
        ax = axes[-1]
        # tune whether to vectorize
        key = stage_name + '_vectorize'
        ctx.knob_manager.define_value(key, 0, 1, 0, True) 
        do_vec = ctx.knob_manager.get_val(key)
        if do_vec:
            vectorize(ctx, stage, ax)

class mergeConsumerOp(schedOp):
    def perform(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        consumers = getConsumers(s, stage_name)
        assert len(consumers) == 1
        consumer = consumers[0]
        c_stage = mapNametoStage(s, consumer)
        # Define compute position
        axes = c_stage.leaf_iter_vars
        n_axes = len(axes)
        group_size = min(n_axes, 5)
        if len(axes) > 5:
            key = stage_name + '_gid_' + self.name
            ctx.knob_manager.define_value(key, 0, n_axes // 5, 0, True)
            gidx = ctx.knob_manager.get_val(key)
        else:
            gidx = 0
        key = stage_name + '_goff_' + self.name
        ctx.knob_manager.define_value(key, 0, 4, 0, True)
        goff = ctx.knob_manager.get_val(key)
        idx = gidx * 5 + goff
        ax = axes[min(idx, len(axes) - 1)]
        compute_at(ctx, stage, c_stage, ax)
        to_vec = stage.leaf_iter_vars[-1]
        vectorize(ctx, stage, to_vec)

class tensorizeOp(schedOp):
    def perform(self, s, stage_name, ctx):
        info = ctx.tensorize_info
        stage = mapNametoStage(s, stage_name)
        intrinsic, align_i, align_k = info['intrin']
        _, index = info['com']
        spatial_idx, reduce_idx = index
        spatial_, reduce_ = self.findUnscheduledAxes(ctx, stage_name, stage.leaf_iter_vars)
        # Tile keys for all spatial axis
        keys = []
        for idx, ax in enumerate(spatial_):
            if idx != spatial_idx:
                key = 1
            else:
                key = align_i
            keys.append(key)
        s_outer, s_inner = self.tileUnderKeys(stage, spatial_, keys, True, ctx)
        outer_names = [x.var.name for x in s_outer]
        ctx.addSTileStucture(stage_name, outer_names, "tensorize")
        # Tile keys for all reduce axis
        keys = []
        for idx, ax in enumerate(reduce_):
            if idx != reduce_idx:
                key = 1
            else:
                key = align_k
            keys.append(key)
        r_outer, r_inner = self.tileUnderKeys(stage, reduce_, keys, True, ctx)
        outer_names = [x.var.name for x in r_outer]
        ctx.addRTileStucture(stage_name, outer_names, "tensorize")

        reorder(ctx, stage, s_outer + r_outer + s_inner + r_inner)
        tensorize_ax = s_inner[spatial_idx]
        tensorize_x86(ctx, stage, tensorize_ax, intrinsic)



class CPUfinishOp(finishOp):
    def update_compute_at_candidates(self, s, stage_name, ctx):
        if stage_name not in ctx.compute_pos_names.keys():
            return
        unroll_pos_idx = None; tag_dict = {}
        if stage_name in ctx.stile_structures.keys(): 
            tups = ctx.stile_structures[stage_name]
            for i, tup in enumerate(tups):
                _, tag = tup
                if tag == "unroll":
                    unroll_pos_idx = i
                tag_dict[tag] = i
        global_pos_key = stage_name + "_global_pos"
        if global_pos_key in ctx.knob_manager.knob_names:
            extent = len(ctx.compute_pos_names[stage_name]) - 1
            up = ctx.knob_manager.solver.vals[global_pos_key].up
            up = min(up, extent)
            ctx.knob_manager.solver.vals[global_pos_key].up = up

all_op_methods = {
   "start" : startOp,
   "finish" : CPUfinishOp,
   "mergeConsumer" : mergeConsumerOp,
   "tileSpatial" : TileSpatialOp,
   "tileAll" : TileAllOp,
   "generalTile" : generalTileOp,
   "vectorize" : CPUvectorizeOp,
   "parallel" : parallelOp,
   "unrollPragma" : unrollPragmaOp,
   "fuseAll" : fuseAllOp,
   "computeAt" : computeAtOp,
   "compute_inline" : InlineOp,
   "tileForCache" : tileForCacheOp,
   "tensorize" : tensorizeOp,
   "addCacheWriteGlobal" : addCacheWriteGlobalOp,
   }

def do_unroll(s, stage_name, ctx):
    if stage_name in ctx.unrolled_stages:
        return False
    # Should be root stage
    if not isRootStage(s, stage_name, ctx):
        return False
    return True

def do_parallel(s, stage_name, ctx):
    if stage_name in ctx.parallel_stages:
        return False
    # Should be root stage
    if not isRootStage(s, stage_name, ctx):
        return False
    return True

def do_generaltile(s, stage_name, ctx):
    # Should have data reuse
    if not hasDataReuse(s, stage_name):
        return False
    ctx.general_tile_stages.append(stage_name)
    return True

def do_tensorize(s, stage_name, ctx):
    if ctx.tensorize_info == None:
        return False
    info = ctx.tensorize_info
    assert 'x86_dot' in info['name']
    t_stage_name, _ = info['com']
    if stage_name != t_stage_name:
        return False
    return True

def do_vectorize(s, stage_name, ctx):
    # Should not be rensorized
    if stage_name in ctx.tensorized_stages:
        return False
    # Should not have reduce axis
    stage = mapNametoStage(s, stage_name)
    if hasattr(stage.op, 'reduce_axis') and len(stage.op.reduce_axis) > 0:
        return False
    if len(stage.leaf_iter_vars) == 0:
        return False
    ctx.vectorized_stages.append(stage_name)
    return True

def do_tile_for_cache(s, stage_name, ctx):
    if stage_name not in ctx.cached_stages:
        return False
    return True


def orgnize_stages(s, ctx):
    if ctx.stage_orgnize == None:
        return
    if ctx.stage_orgnize == 'X86Tensorize_format':
        tups = ctx.tensorize_info['stage_orgination']
        for tup in tups:
            ctx.addSched(tup[1], tup[0], s)

def do_merge_consumer(s, stage_name, ctx):
    if 'pad' in stage_name:
        return True
    return False

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
        if do_merge_consumer(s, stage_name, ctx):
            ctx.addSched('mergeConsumer', stage_name, s)
            # No further schedule
            continue
        if do_unroll(s, stage_name, ctx):
            ctx.addSched('unrollPragma', stage_name, s)
        if do_parallel(s, stage_name, ctx):
            ctx.addSched("parallel", stage_name, s)
        if do_tile_for_cache(s, stage_name, ctx):
            ctx.addSched('tileForCache', stage_name, s)
        if do_tensorize(s, stage_name, ctx):
            ctx.addSched('tensorize', stage_name, s)
        if do_generaltile(s, stage_name, ctx):
            ctx.addSched('generalTile', stage_name, s)
        if do_vectorize(s, stage_name, ctx):
            ctx.addSched('vectorize', stage_name, s)
        ctx.addSched("finish", stage_name, s)

