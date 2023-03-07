import tvm
import tvm.te as te
from Heron.utils import *
from Heron.schedule.primitives import *
from .sched_common import *
from .gpu_common import *
from .ana_common import *
class tileThreadOp(tileBindOp):
    def __init__(self, name):
        self.thread_type = "threadIdx.x"
        self.name = name

    def check(self, s, stage_name, ctx):
        axes = self.formTobindAxes(s, stage_name, ctx)
        return axes

    def addBinded(self, stage_name, ctx):
        ctx.bind_thread_stages.append(stage_name)

class tileVThreadOp(tileBindOp):
    def __init__(self, name):
        self.thread_type = "vthread"
        self.name = name

    def check(self, s, stage_name, ctx):
        axes = self.formTobindAxes(s, stage_name, ctx)
        return axes

    def addBinded(self, stage_name, ctx):
        ctx.bind_vthread_stages.append(stage_name)

class tileBlockAndThreadOp(schedOp):
    def perform(self, s, stage_name, ctx):
        block_binder = tileBlockOp('tileBlock')
        vthread_binder = tileVThreadOp('tileVThread')
        thread_binder = tileThreadOp('tileThread')
        block_binder.perform(s, stage_name, ctx)
        vthread_binder.perform(s, stage_name, ctx)
        thread_binder.perform(s, stage_name, ctx)

class defaultSharedLoadSchedOp(schedOp):
    def perform(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        # Fuse all
        fuseall_op = fuseAllOp('fuseAll')
        fused = fuseall_op.perform(s, stage_name, ctx)
        # vectorize
        key = stage_name + "_vectorize"
        ctx.knob_manager.define_value(key, 1, 4, 1, True)
        ctx.knob_manager.addCandidates(key, [1, 2, 4])
        p = ctx.knob_manager.get_val(key)
        a_o, a_i = split(ctx, stage, fused, key, factor = p, update_dep_graph = False)
        vectorize(ctx, stage, a_i)
        # bind thread
        thread_num = ctx.knob_manager.get_val("threadIdx.x")
        a_o_o, a_o_i = split(ctx, stage, a_o, "threadIdx.x", factor = thread_num, update_dep_graph = False)
        bind(ctx, stage, a_o_i, "threadIdx.x")
        ctx.bind_thread_stages.append(stage_name)

class GPUvectorizeOp(schedOp):
    def check(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        ax = stage.leaf_iter_vars[-1]
        return stage, ax

    def perform(self, s, stage_name, ctx):
        stage, ax = self.check(s, stage_name, ctx)
        ctx.addSchedDesc("\n## Vectorize \n", True)
        knob_key = stage_name + "_vectorize"
        ctx.knob_manager.define_value(knob_key, 1, 4, 1, True)
        ctx.knob_manager.addCandidates(knob_key, [1, 2, 4])
        tile_op = TileSpatialOp("tileSpatial")
        outer, inner, keys = tile_op.perform(s, stage_name, ctx)
        fused = fuse(ctx, stage, inner)
        vec_key = stage_name + '_' + fused.var.name
        ctx.knob_manager.addEQ(vec_key, knob_key)
        vectorize(ctx, stage, fused)


all_op_methods = {
   "start" : startOp,
   "finish" : GPUfinishOp,
   "skip": skipOp,
   "tileSpatial" : TileSpatialOp,
   "tileAll" : TileAllOp,
   "generalTile" : generalTileOp,
   "vectorize" : GPUvectorizeOp,
   "unrollPragma" : unrollPragmaOp,
   "tileBlock" : tileBlockOp,
   "tileBlockAndThread" : tileBlockAndThreadOp,
   "tileVThread" : tileVThreadOp,
   "fuseAll" : fuseAllOp,
   "defaultSharedLoadSched" : defaultSharedLoadSchedOp,
   "computeAt" : computeAtOp,
   "compute_inline" : InlineOp,
   "addCacheReadShared" : addCacheReadSharedOp,
   "addCacheWriteLocal" : addCacheWriteLocalOp,
   "addRfactor" : addRfactorOp,
   }

def orgnize_stages(s, ctx):
    stage_ordered = getStageNamesOrdered(s)
    for stage_name in stage_ordered:
        if stage_name in ctx.no_schedule_stages:
            continue
        if hasMoreReduce(s, stage_name):
            ctx.addSched("addRfactor", stage_name, s)
            continue
        if not hasDataReuse(s, stage_name):
            continue
        ctx.addSched("addCacheWriteLocal", stage_name, s)
        ctx.addSched("addCacheReadShared", stage_name + '.local', s)

def sched_via_rule(ctx, s):
    # First modify stages
    ctx.InlineAll(s)
    ctx.updateAxisLength(s)
    orgnize_stages(s, ctx)
    ctx.updateAxisLength(s)
    # The schedule each stage in topological order
    stage_ordered = getStageNamesOrdered(s)
    for stage_name in stage_ordered:
        if stage_name in ctx.no_schedule_stages:
            continue
        ctx.addSched("start", stage_name, s)
        if do_shareload_default_sched(s, stage_name, ctx):
            ctx.addSched("defaultSharedLoadSched", stage_name, s)
            continue

        # Apply schedule to stage according to static analysis   
        if do_unroll(s, stage_name, ctx):
            ctx.addSched('unrollPragma', stage_name, s)

        if do_bindblock(s, stage_name, ctx):
            ctx.addSched("tileBlockAndThread", stage_name, s)

        if do_generaltile(s, stage_name, ctx):
            ctx.addSched('generalTile', stage_name, s)

        if do_vectorize(s, stage_name, ctx):
            ctx.addSched('vectorize', stage_name, s)
        ctx.addSched("finish", stage_name, s)

