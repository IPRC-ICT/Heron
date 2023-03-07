import tvm
import tvm.te as te
from Heron.utils import *
from Heron.schedule.primitives import *
from .sched_common import *
from .gpu_common import *
from .ana_common import *

class GPUvectorizeOp(schedOp):
    def check(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        ax = stage.leaf_iter_vars[-1]
        return stage, ax

    def perform(self, s, stage_name, ctx):
        stage, ax = self.check(s, stage_name, ctx)
        ctx.addSchedDesc("\n## Vectorize \n")
        knob_key = stage_name + "_vectorize"
        ctx.knob_manager.define_value(knob_key, 1, 8, 1, True)
        ctx.knob_manager.addCandidates(knob_key, [1, 2, 4, 8])
        tile_op = TileSpatialOp("tileSpatial")
        outer, inner, keys = tile_op.perform(s, stage_name, ctx)
        fused = fuse(ctx, stage, inner)
        vec_key = stage_name + '_' + fused.var.name
        ctx.knob_manager.addEQ(vec_key, knob_key)
        vectorize(ctx, stage, fused)

class tileThreadOp(tileBindOp):
    def __init__(self, name):
        self.thread_type = "threadIdx.y"
        self.name = name

    def check(self, s, stage_name, ctx):
        axes = self.formTobindAxes(s, stage_name, ctx)
        return axes

    def addBinded(self, stage_name, ctx):
        ctx.bind_thread_stages.append(stage_name)

class tileWarpOp(tileBindOp):
    def __init__(self, name):
        self.thread_type = "threadIdx.x"
        self.name = name

    def check(self, s, stage_name, ctx):
        axes = self.formTobindAxes(s, stage_name, ctx)
        return axes

    def addBinded(self, stage_name, ctx):
        ctx.bind_warp_stages.append(stage_name)

class defaultSchedOp(schedOp):
    def checkAndDefine(self, name, ctx):
        limit = ctx.getThreadLimit(name)
        defined =  name in ctx.knob_manager.solver.vals.keys()
        if not defined:
            ctx.knob_manager.define_value(name, 1, limit, 1)


    def perform(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        # Fuse, bind_block, bind_thread, vectorize
        # Fuse all
        fuseall_op = fuseAllOp('fuseAll')
        fused = fuseall_op.perform(s, stage_name, ctx, False)

        # vectorize
        key = stage_name + "_vectorize"
        ctx.knob_manager.define_value(key, 1, 8, 1, True)
        ctx.knob_manager.addCandidates(key, [1, 2, 4, 8])
        p = ctx.knob_manager.get_val(key)
        a_o, a_i = split(ctx, stage, fused, key, factor = p, update_dep_graph = False)
        vectorize(ctx, stage, a_i)

        self.checkAndDefine("threadIdx.x", ctx)
        self.checkAndDefine("threadIdx.y", ctx)
        self.checkAndDefine("blockIdx.x", ctx)
        # bind thread
        thread_num = ctx.knob_manager.get_val("threadIdx.x")
        a_o_o, a_o_i = split(ctx, stage, a_o, "threadIdx.x", factor = thread_num, update_dep_graph = False)
        bind(ctx, stage, a_o_i, "threadIdx.x")
        # bind thread
        if "threadIdx.y" in ctx.knob_manager.solver.vals.keys():
            thread_num = ctx.knob_manager.get_val("threadIdx.y")
            a_o_o_o, a_o_o_i = split(ctx, stage, a_o_o, "threadIdx.y", factor = thread_num, update_dep_graph = False)
            bind(ctx, stage, a_o_o_i, "threadIdx.y")
            ctx.bind_thread_stages.append(stage_name)
        else:
            a_o_o_o = a_o_o
        bind(ctx, stage, a_o_o_o, "blockIdx.x")

class defaultSharedLoadSchedOp(schedOp):
    def perform(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        # Storage align
        storage_align_op = storageAlignOp('storage_align')
        storage_align_op.perform(s, stage_name, ctx)
        # Fuse all
        fuseall_op = fuseAllOp('fuseAll')
        fused = fuseall_op.perform(s, stage_name, ctx)
        # vectorize
        key = stage_name + "_vectorize"
        ctx.knob_manager.define_value(key, 1, 8, 1, True)
        ctx.knob_manager.addCandidates(key, [1, 2, 4, 8])
        p = ctx.knob_manager.get_val(key)
        a_o, a_i = split(ctx, stage, fused, key, factor = p, update_dep_graph = False)
        vectorize(ctx, stage, a_i)
        # bind thread
        thread_num = ctx.knob_manager.get_val("threadIdx.x")
        a_o_o, a_o_i = split(ctx, stage, a_o, "threadIdx.x", factor = thread_num, update_dep_graph = False)
        bind(ctx, stage, a_o_i, "threadIdx.x")
        ctx.bind_warp_stages.append(stage_name)
        # bind thread
        if "threadIdx.y" in ctx.knob_manager.solver.vals.keys():
            thread_num = ctx.knob_manager.get_val("threadIdx.y")
            a_o_o_o, a_o_o_i = split(ctx, stage, a_o_o, "threadIdx.y", factor = thread_num, update_dep_graph = False)
            bind(ctx, stage, a_o_o_i, "threadIdx.y")
            ctx.bind_thread_stages.append(stage_name)

class tensorcoreLoadAOp(schedOp):
    def perform(self, s, stage_name, ctx):
        ctx.addSchedDesc("\n## Tensor core loadA\n")
        stage = mapNametoStage(s, stage_name)
        shape = genTCShape(ctx)
        src_strides, dst_strides, a_shape, tensorize_ax_idx, wmma_m_idx, wmma_k_idx, dtype = genTCLoadAParams(ctx, shape) 

        spatial_, reduce_ = self.findUnscheduledAxes(ctx, stage_name, stage.leaf_iter_vars)
        # Tile keys for all spatial axis
        keys = []
        for idx, ax in enumerate(spatial_):
            if idx not in (wmma_k_idx, wmma_m_idx):
                key = 1
            elif idx == wmma_k_idx:
                key = "wmma_k"
            else:
                key = "wmma_m"
            keys.append(key)
        outer, inner = self.tileUnderKeys(stage, spatial_, keys, True, ctx)
        outer_names = [x.var.name for x in outer]
        ctx.addSTileStucture(stage_name, outer_names, "tensorize")
        reorder(ctx, stage, outer + inner)

        layout = ctx.tensorize_info['loadA'][-2]
        tensorize_ax = inner[tensorize_ax_idx]
        tensorize(ctx, stage, tensorize_ax, 'intrin_wmma_load_matrix_A', \
                                        (dst_strides, src_strides,\
                                        shape, layout, a_shape, a_shape, dtype))

class tensorcoreLoadBOp(schedOp):
    def perform(self, s, stage_name, ctx):
        ctx.addSchedDesc("\n## Tensor core loadB\n")
        stage = mapNametoStage(s, stage_name)
        shape = genTCShape(ctx)
        src_strides, dst_strides, b_shape, tensorize_ax_idx, wmma_k_idx, wmma_n_idx, dtype = genTCLoadBParams(ctx, shape) 

        spatial_, reduce_ = self.findUnscheduledAxes(ctx, stage_name, stage.leaf_iter_vars)
        # Tile keys for all spatial axis
        keys = []
        for idx, ax in enumerate(spatial_):
            if idx not in (wmma_k_idx, wmma_n_idx):
                key = 1
            elif idx == wmma_k_idx:
                key = "wmma_k"
            else:
                key = "wmma_n"
            keys.append(key)
        outer, inner = self.tileUnderKeys(stage, spatial_, keys, True, ctx)
        outer_names = [x.var.name for x in outer]
        ctx.addSTileStucture(stage_name, outer_names, "tensorize")
        reorder(ctx, stage, outer + inner)

        tensorize_ax = inner[tensorize_ax_idx]
        layout = ctx.tensorize_info['loadB'][-2]
        tensorize(ctx, stage, tensorize_ax, 'intrin_wmma_load_matrix_W', \
                                        (dst_strides, src_strides,\
                                        shape, layout, b_shape, b_shape, dtype))

class tensorcoreComputeOp(schedOp):
    def perform(self, s, stage_name, ctx):
        ctx.addSchedDesc("\n## Tensor core compute\n")
        stage = mapNametoStage(s, stage_name)
        shape = genTCShape(ctx)
        _, a_strides, a_shape, _, _, _, _ = genTCLoadAParams(ctx, shape) 
        _, b_strides, b_shape, _, _, _, _ = genTCLoadBParams(ctx, shape) 
        c_strides, _, c_shape, _, _, _, _ = genTCStoreParams(ctx, shape) 
        (wmma_m_idx, wmma_k_idx, wmma_n_idx), dtype = ctx.tensorize_info['com']

        spatial_, reduce_ = self.findUnscheduledAxes(ctx, stage_name, stage.leaf_iter_vars)
        # Tile keys for all spatial axis
        keys = []
        for idx, ax in enumerate(spatial_):
            if idx not in (wmma_n_idx, wmma_m_idx):
                key = 1
            elif idx == wmma_m_idx:
                key = "wmma_m"
            else:
                key = "wmma_n"
            keys.append(key)
        s_outer, s_inner = self.tileUnderKeys(stage, spatial_, keys, True, ctx)
        outer_names = [x.var.name for x in s_outer]
        ctx.addSTileStucture(stage_name, outer_names, "tensorize")
        # Tile keys for all reduce axis
        keys = []
        for idx, ax in enumerate(reduce_):
            if idx != wmma_k_idx:
                key = 1 
            else:
                key = "wmma_k"
            keys.append(key)
        r_outer, r_inner = self.tileUnderKeys(stage, reduce_, keys, True, ctx)
        outer_names = [x.var.name for x in r_outer]
        ctx.addRTileStucture(stage_name, outer_names, "tensorize")

        reorder(ctx, stage, s_outer + r_outer + s_inner + r_inner)

        tensorize_ax = s_inner[wmma_m_idx]
        indtype = ctx.tensorize_info['loadA'][-1]
        a_gemm, b_gemm, c_gemm = ctx.tensorize_info['compute_func'](c_shape, a_shape, b_shape, indtype, dtype)
        tensorize(ctx, stage, tensorize_ax, 'intrin_wmma_gemm', \
                                (a_gemm, b_gemm, c_gemm,\
                                 a_strides, b_strides, c_strides, shape))

class tensorcoreStoreOp(schedOp):
    def perform(self, s, stage_name, ctx):
        ctx.addSchedDesc("\n## Tensor core store\n")
        stage = mapNametoStage(s, stage_name)
        shape = genTCShape(ctx)
        src_strides, dst_strides, c_shape, tensorize_ax_idx, wmma_m_idx, wmma_n_idx, dtype = genTCStoreParams(ctx, shape) 

        spatial_, reduce_ = self.findUnscheduledAxes(ctx, stage_name, stage.leaf_iter_vars)
        # Tile keys for all spatial axis
        keys = []
        for idx, ax in enumerate(spatial_):
            if idx not in (wmma_m_idx, wmma_n_idx):
                key = 1
            elif idx == wmma_m_idx:
                key = "wmma_m"
            else:
                key = "wmma_n"
            keys.append(key)
        outer, inner = self.tileUnderKeys(stage, spatial_, keys, True, ctx)
        outer_names = [x.var.name for x in outer]
        ctx.addSTileStucture(stage_name, outer_names, "tensorize")
        reorder(ctx, stage, outer + inner)

        tensorize_ax = inner[tensorize_ax_idx]
        tensorize(ctx, stage, tensorize_ax, 'intrin_wmma_store_matrix', \
                                        (dst_strides, src_strides,\
                                        shape, dtype, c_shape, c_shape))

class addCacheTensorCoreOp(schedOp):
    def check(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        inputs = stage.op.input_tensors
        input_tensors = []
        for inp in inputs:
            input_tensors.append(ctx.tensor_dict[inp.name])
        return input_tensors

    def perform(self, s, stage_name, ctx):
        inputs = self.check(s, stage_name, ctx)
        ctx.addSchedDesc("\n## Cache Tensor Core\n")
        wmma_names = ["wmma.matrix_a", "wmma.matrix_b"]
        # Define knobs for wmma_m, wmma_n
        keym = "wmma_m"; keyk = "wmma_k"; keyn = "wmma_n"
        ctx.knob_manager.define_value(keym, 8, 32, 16, True)
        ctx.knob_manager.addCandidates(keym, [8, 16, 32])

        ctx.knob_manager.define_value(keyk, 16, 16, 16)

        ctx.knob_manager.define_value(keyn, 8, 32, 16, True)
        ctx.knob_manager.addCandidates(keyn, [8, 16, 32])

        keys = [keym, keyn, keyk]
        ctx.knob_manager.addProd(keys, 16 * 16 * 16)

        out = ctx.tensor_dict[stage_name]
        addcacheshared_op = addCacheReadSharedOp("addCacheReadShared")
        # Cache write for wmma units
        out_wmma = cache_write(ctx, out, "wmma.accumulator", s)
        shares = addcacheshared_op.perform(s, out_wmma.name, ctx)
        # Cache read shared for operands
        for idx, inp in enumerate(inputs):
            shared = shares[idx]
            in_wmma = cache_read(ctx, shared, wmma_names[idx], out_wmma, s)
            self.define_com_pos(in_wmma.name, s[out_wmma], 'local_pos', ctx)
            # Tensorize for load
            if inp.name == ctx.tensorize_info['loadA'][0]:
                ctx.tensorize_loadA_stage = in_wmma.name
            elif inp.name == ctx.tensorize_info['loadB'][0]:
                ctx.tensorize_loadB_stage = in_wmma.name
            else:
                raise ValueError("%s is expected to be either\
                        tensorcore \'loadA\' or \'loadB\'"%inp.name)
        # Cache read shared
        out_shares = addcacheshared_op.perform(s, out.name, ctx)
        out_shared = out_shares[0]
        self.define_com_pos(out_wmma.name, s[out_shared], 'local_pos', ctx)
        # Tensorize for compute
        ctx.tensorize_com_stage = out_wmma.name
        # Tensorize for store
        ctx.tensorize_store_stage = out_shared.name
        # Restrict wmma to compute at threadidx.y tag if exists
        ctx.pos_via_tag[out_shared.name + "_" + "local_pos"] = "threadIdx.y"
        # Restrict shared to compute at blockidx.x tag if exists
        ctx.pos_via_tag[out.name + "_" + "shared_pos"] = "blockIdx.x"

all_op_methods = {
   # Form schedule
   "start" : TCStartOp,
   "finish" : GPUfinishOp,
   # Normal schedule
   "skip": skipOp,
   "tileSpatial" : TileSpatialOp,
   "tileAll" : TileAllOp,
   "generalTile" : generalTileOp,
   "vectorize" : GPUvectorizeOp,
   "unrollPragma" : unrollPragmaOp,
   "storage_align" : storageAlignOp,
   "defaultGPUSched" : defaultSchedOp, 
   # For GPU bind
   "tileBlock" : tileBlockOp,
   "tileThread" : tileThreadOp,
   "tileWarp" : tileWarpOp, 
   "fuseAll" : fuseAllOp,
   "defaultSharedLoadSched" : defaultSharedLoadSchedOp,
   "computeAt" : computeAtOp,
   "compute_inline" : InlineOp,
   "tensorcoreLoadA" : tensorcoreLoadAOp,
   "tensorcoreLoadB" : tensorcoreLoadBOp,
   "tensorcoreCompute" : tensorcoreComputeOp,
   "tensorcoreStore" : tensorcoreStoreOp,
   "addCacheReadShared" : addCacheReadSharedOp,
   "addCacheTensorCore" : addCacheTensorCoreOp,
   }


def orgnize_stages(s, ctx):
    assert ctx.stage_orgnize == 'Tensorcore_format'
    stage_name = ctx.tensorize_info['stage_name']
    ctx.addSched("addCacheTensorCore", stage_name, s)

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

        if do_default_sched(s, stage_name, ctx):
            ctx.addSched("defaultGPUSched", stage_name, s)
            continue

        # Apply schedule to stage according to static analysis   
        if do_unroll(s, stage_name, ctx):
            ctx.addSched('unrollPragma', stage_name, s)

        if do_storage_align(s, stage_name, ctx):
            ctx.addSched('storage_align', stage_name, s)

        if do_bindblock(s, stage_name, ctx):
            ctx.addSched("tileBlock", stage_name, s)

        if do_bindthread(s, stage_name, ctx):
            ctx.addSched("tileThread", stage_name, s)

        if do_bindwarp(s, stage_name, ctx):
            ctx.addSched("tileWarp", stage_name, s)

        if do_generaltile(s, stage_name, ctx):
            ctx.addSched('generalTile', stage_name, s)

        if stage_name == ctx.tensorize_loadA_stage:
            ctx.addSched('tensorcoreLoadA', stage_name, s)
        elif stage_name == ctx.tensorize_loadB_stage:
            ctx.addSched('tensorcoreLoadB', stage_name, s)
        elif stage_name == ctx.tensorize_com_stage:
            ctx.addSched('tensorcoreCompute', stage_name, s)
        elif stage_name == ctx.tensorize_store_stage:
            ctx.addSched('tensorcoreStore', stage_name, s)

        if do_vectorize(s, stage_name, ctx):
            ctx.addSched('vectorize', stage_name, s)
        ctx.addSched("finish", stage_name, s)
    ctx.addThreadLimit(s)
    ctx.addShareMemLimit(s)

