import tvm
import tvm.te as te
from Heron.utils import *
from Heron.schedule.primitives import *
from .sched_common import *
from .ana_common import *

class TCStartOp(startOp):
    def fixAxesLength(self, s, stage_name, ctx):
        if stage_name not in ctx.shared_load_stages:
            return {}
        attach_name = ctx.compute_poses[stage_name][0]
        a_stage = mapNametoStage(s, attach_name)
        if len(a_stage.op.input_tensors) == 1:
            return {}
        coe = {}
        stage = mapNametoStage(s, stage_name)
        for idx, ax in enumerate(stage.op.axis):
            keys = ctx.stage_warp_nums[ctx.compute_poses[attach_name][0]]
            m = ctx.axis_map[stage_name][idx]
            _type, _idx = m.split('_')
            if _type == 'r':
                continue
            key = keys[int(_idx)]
            ax_key = genKey("L", stage_name, str(ax.var.name))
            coe[ax_key] = key
        return coe

class tileBindOp(schedOp):
    def claim_knobs(self, ctx, stage_name, keys):
        limit = ctx.getThreadLimit(self.thread_type)
        defined = self.thread_type in ctx.knob_manager.solver.vals.keys()
        if not defined:
            ctx.knob_manager.define_value(self.thread_type, 1, limit, 1)
        ctx.knob_manager.addProd(keys, self.thread_type)
        if self.thread_type == "threadIdx.x" and ctx.is_tensorcore:
            ctx.knob_manager.addEQ(self.thread_type, 32)
        if self.thread_type == "threadIdx.y" and ctx.is_tensorcore:
            ctx.stage_warp_nums[stage_name] = keys

    def restrict4tc(self, keys, stage, ctx, s):
        if not ctx.knob_manager.is_building:
            return
        if not ctx.is_tensorcore:
            return
        if self.thread_type == "blockIdx.x":
            output_names = [x.name for x in s.outputs]
            if stage.op.name not in output_names:
                return
            c_m_idx, c_n_idx = ctx.tensorize_info['store'][0]
            for idx, key in enumerate(keys):
                if idx in (c_m_idx, c_n_idx):
                    continue
                else:
                    # Restrict to maximum, do not let these axes appear in thread level
                    up = ctx.knob_manager.solver.vals[key].up
                    ctx.knob_manager.addEQ(key, up)

    def perform(self, s, stage_name, ctx):
        ctx.addSchedDesc("\n## Bind %s\n"%self.thread_type, True)
        stage = mapNametoStage(s, stage_name)
        tile_op = TileSpatialOp("tileSpatial")
        tile_op.tag = self.thread_type
        outer, inner, keys = tile_op.perform(s, stage_name, ctx)
        self.claim_knobs(ctx, stage_name, keys)
        self.restrict4tc(keys, stage, ctx, s)
        # fuse
        fused = fuse(ctx, stage, outer)
        # bind
        bind(ctx, stage, fused, self.thread_type)
        self.addBinded(stage_name, ctx)
        return inner


class tileBlockOp(tileBindOp):
    def __init__(self, name):
        self.thread_type = "blockIdx.x"
        self.name = name

    def check(self, s, stage_name, ctx):
        axes = self.formTobindAxes(s, stage_name, ctx)
        return axes

    def addBinded(self, stage_name, ctx):
        ctx.bind_block_stages.append(stage_name)

class tileBlockAndThreadOp(schedOp):
    def perform(self, s, stage_name, ctx):
        block_binder = tileBlockOp('tileBlock')
        vthread_binder = tileVThreadOp('tileVThread')
        thread_binder = tileThreadOp('tileThread')
        block_binder.perform(s, stage_name, ctx)
        vthread_binder.perform(s, stage_name, ctx)
        thread_binder.perform(s, stage_name, ctx)

class addCacheReadSharedOp(schedOp):
    def check(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        inputs = stage.op.input_tensors
        input_tensors = []
        for inp in inputs:
            input_tensors.append(ctx.tensor_dict[inp.name])
        return input_tensors

    def perform(self, s, stage_name, ctx):
        inputs = self.check(s, stage_name, ctx)
        ctx.addSchedDesc("\n## Cache read shared\n", True)
        stage = mapNametoStage(s, stage_name)
        stage_tensor = ctx.tensor_dict[stage_name]
        outs = []
        for inp in inputs:
            cached = cache_read(ctx, inp, 'shared', stage_tensor, s)
            self.define_com_pos(cached.name, s[stage_tensor], 'shared_pos', ctx)
            ctx.shared_load_stages.append(cached.name)
            outs.append(cached)
        consumer = hasFusibleConsumer(s, outs[0].name)
        if consumer != None:
            return outs
        else:
            # If cache read inputs, do storage_align, fuseThread, vectorize
            for out in outs:
                ctx.default_sharedload_stages.append(out.name)
        return outs

class addCacheWriteLocalOp(schedOp):
    def perform(self, s, stage_name, ctx):
        ctx.addSchedDesc("\n## Cache write local\n", True)
        stage = mapNametoStage(s, stage_name)
        out = ctx.tensor_dict[stage_name]
        out_chached = cache_write(ctx, out, "local", s)
        self.define_com_pos(out_chached.name, s[out], 'local_pos', ctx)

class storageAlignOp(schedOp):
    def perform(self, s, stage_name, ctx):
        ctx.addSchedDesc("\n## Storage align \n")
        stage = mapNametoStage(s, stage_name)
        ax = stage.op.axis[-1]
        # Factor
        fkey = genKey("L", stage_name, str(ax.var.name))
        factor = ctx.knob_manager.get_val(fkey)
        # Offset
        okey = genKey("P", stage_name, param_name = "offset")
        ctx.knob_manager.define_value(okey, 0, 48, 0, True)
        ctx.knob_manager.addCandidates(okey, [0, 8, 16, 24, 32, 48])
        offset = ctx.knob_manager.get_val(okey)
        # Align size
        key = genKey("P", stage_name, param_name = "align_size")
        ctx.knob_manager.define_value(key, 1, 88888888, 1)
        ctx.knob_manager.addSUM(key, [fkey, okey])
        ctx.align_sizes[stage_name] = key
        align_sz = factor + offset
        storage_align(ctx, stage, stage.leaf_iter_vars[-2], align_sz - 1, align_sz)

class GPUfinishOp(finishOp):
    def update_compute_at_candidates(self, s, stage_name, ctx):
        if stage_name not in ctx.compute_pos_names.keys():
            return
        block_pos_idx = None; warp_pos_idx = None; tensorize_pos_idx = None
        unroll_pos_idx = None;  thread_pos_idx = None; tag_dict = {}
        if stage_name in ctx.stile_structures.keys(): 
            tups = ctx.stile_structures[stage_name]
            for i, tup in enumerate(tups):
                _, tag = tup
                if tag == "blockIdx.x":
                    block_pos_idx = i
                elif tag == "threadIdx.x":
                    thread_pos_idx = i
                elif tag == "threadidx.y":
                    warp_pos_idx
                elif tag == "tensorize":
                    tensorize_pos_idx = i
                elif tag == 'unroll':
                    unroll_pos_idx = i
                tag_dict[tag] = i
        pos_extent = len(ctx.compute_pos_names[stage_name]) - 1
        assert pos_extent > 0
        # Shared pos
        shared_pos_key = genKey("P", stage_name, param_name = "shared_pos")
        if shared_pos_key in ctx.knob_manager.knob_names:
            low = ctx.knob_manager.solver.vals[shared_pos_key].low 
            up = ctx.knob_manager.solver.vals[shared_pos_key].up 
            up = min(up, pos_extent)
            # Shared pos should be after "blockIdx.x" tag
            if block_pos_idx != None:
                low = max(block_pos_idx, low)
            if tensorize_pos_idx != None:
                up = min(tensorize_pos_idx, up)
            if unroll_pos_idx != None:
                ctx.knob_manager.addNE(shared_pos_key, unroll_pos_idx)
            ctx.knob_manager.solver.vals[shared_pos_key].low = low
            ctx.knob_manager.solver.vals[shared_pos_key].up = up
            if shared_pos_key in ctx.pos_via_tag.keys():
                tag = ctx.pos_via_tag[shared_pos_key]
                pos = tag_dict[tag]
                ctx.knob_manager.addEQ(shared_pos_key, pos)

        # Local pos
        local_pos_key = genKey("P", stage_name, param_name = "local_pos")
        if local_pos_key in ctx.knob_manager.knob_names:
            low = ctx.knob_manager.solver.vals[local_pos_key].low 
            up = ctx.knob_manager.solver.vals[local_pos_key].up 
            up = min(up, pos_extent)
            # Local pos should be after "threadIdx" tag
            if warp_pos_idx != None:
                low = max(warp_pos_idx, low)
            if thread_pos_idx != None:
                low = max(thread_pos_idx, low)
            if tensorize_pos_idx != None:
                up = min(tensorize_pos_idx, up)
            if unroll_pos_idx != None:
                ctx.knob_manager.addNE(local_pos_key, unroll_pos_idx)
            ctx.knob_manager.solver.vals[local_pos_key].low = low
            ctx.knob_manager.solver.vals[local_pos_key].up = up
            if local_pos_key in ctx.pos_via_tag.keys():
                tag = ctx.pos_via_tag[local_pos_key]
                pos = tag_dict[tag]
                ctx.knob_manager.addEQ(local_pos_key, pos)

        # If stage has both local pos and shared pos, add resctriction:
        # shared pos < local pos
        if shared_pos_key in ctx.knob_manager.knob_names and \
            local_pos_key in ctx.knob_manager.knob_names:
            ctx.knob_manager.addLE(shared_pos_key, local_pos_key)


def isShareMemAccess(s, stage_name, ctx):
    if stage_name in ctx.shared_load_stages:
        return True
    return False

def do_shareload_default_sched(s, stage_name, ctx):
    if stage_name in ctx.default_sharedload_stages:
        return True
    return False

def do_default_sched(s, stage_name, ctx):
    stage = mapNametoStage(s, stage_name)
    if 'default' in stage.op.tag:
        return True
    else:
        return False

def do_unroll(s, stage_name, ctx):
    # Should be root stage
    if not isRootStage(s, stage_name, ctx):
        return False
    ctx.unrolled_stages.append(stage_name)
    return True

def do_storage_align(s, stage_name, ctx):
    if not isShareMemAccess(s, stage_name, ctx):
        return False
    return True

def do_bindblock(s, stage_name, ctx):
    if not isRootStage(s, stage_name, ctx):
        return False
    stage = mapNametoStage(s, stage_name)
    # should have parallel axis
    if not hasattr(stage.op, 'axis') or len(stage.op.axis) == 0:
        return False
    return True

def do_bindthread(s, stage_name, ctx):
    stage = mapNametoStage(s, stage_name)
    # should have parallel axis
    if not hasattr(stage.op, 'axis') or len(stage.op.axis) == 0:
        return False
    if isRootStage(s, stage_name, ctx) or isShareMemAccess(s, stage_name, ctx):
        return True
    return False

def do_bindwarp(s, stage_name, ctx):
    stage = mapNametoStage(s, stage_name)
    if not ctx.is_tensorcore:
        return False
    if stage_name == ctx.tensorize_loadA_stage or\
       stage_name == ctx.tensorize_loadB_stage or\
       stage_name == ctx.tensorize_com_stage or\
       stage_name == ctx.tensorize_store_stage:
        return False

    # should have parallel axis
    if not hasattr(stage.op, 'axis') or len(stage.op.axis) == 0:
        return False
    if isRootStage(s, stage_name, ctx) or isShareMemAccess(s, stage_name, ctx):
        return True
    return False

def do_generaltile(s, stage_name, ctx):
    # Should have data reuse
    if not hasDataReuse(s, stage_name):
        return False
    ctx.general_tile_stages.append(stage_name)
    return True

def do_vectorize(s, stage_name, ctx):
    # Should not be tensorized
    if stage_name in ctx.tensorized_stages:
        return False
    stage = mapNametoStage(s, stage_name)
    # should have parallel axis
    if not hasattr(stage.op, 'axis') or len(stage.op.axis) == 0:
        return False
    if hasattr(stage.op, 'reduce_axis') and len(stage.op.reduce_axis) > 0:
        return False
    ctx.vectorized_stages.append(stage_name)
    return True

