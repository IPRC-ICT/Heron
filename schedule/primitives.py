import tvm.te as te
from tvm.topi.cuda.tensor_intrin import *
from Heron.utils import *

def split(ctx, stage, ax, knob_key, nparts = None, factor = None, update_dep_graph = True):
    ax_key = stage.op.name + '_' + ax.var.name
    if nparts != None:
        axo, axi = stage.split(ax, nparts = nparts)
        strs =  "%s, %s = s[%s].split(%s, nparts = %d)\n"%(
                getAxname(axo), getAxname(axi), getStageName(stage), getAxname(ax), nparts)

    if factor != None:
        axo, axi = stage.split(ax, factor = factor)
        strs =  "%s, %s = s[%s].split(%s, factor = %d)\n"%(
                getAxname(axo), getAxname(axi), getStageName(stage), getAxname(ax), factor)
    ax_key = stage.op.name + '_' + ax.var.name
    axo_key = stage.op.name + '_' + axo.var.name
    axi_key = stage.op.name + '_' + axi.var.name
    ctx.addSchedDesc(strs)
    ctx.knob_manager.updateAxisParents(stage.op.name, axo.var.name, [ax.var.name])
    ctx.knob_manager.updateAxisParents(stage.op.name, axi.var.name, [ax.var.name])
    if stage.op.name in ctx.compute_pos_names.keys():
        poses = ctx.compute_pos_names[stage.op.name]
        if ax.var.name in poses:
            idx = poses.index(ax.var.name)
            front = poses[:idx]
            back = poses[idx+1:]
            ctx.compute_pos_names[stage.op.name] = front + [axo.var.name, axi.var.name] + back
    if nparts != None:
        ctx.scheduled_axes.append(axo_key)
        if update_dep_graph:
            ctx.knob_manager.addSplitNparts(ax_key, axo_key, axi_key, knob_key)
    else:
        ctx.scheduled_axes.append(axi_key)
        if update_dep_graph:
            ctx.knob_manager.addSplitFactor(ax_key, axo_key, axi_key, knob_key)
    return axo, axi

def reorder(ctx, stage, tups):
    stage.reorder(*tups)
    strs = "s[%s].reorder("%(getStageName(stage))
    for ax in tups:
        strs += getAxname(ax) + ', '
    strs += ")\n"
    ctx.addSchedDesc(strs)

def parallel(ctx, stage, ax):
    stage.parallel(ax)
    strs = "s[%s].parallel(%s)\n"%(getStageName(stage), getAxname(ax))
    ctx.addSchedDesc(strs)


def fuse(ctx, stage, tups, update_dep_graph = True):
    fused = stage.fuse(*tups)
    strs = "%s = s[%s].fuse("%(getAxname(fused), getStageName(stage))
    for ax in tups:
        strs += getAxname(ax) + ', '
    strs += ")\n"
    ctx.addSchedDesc(strs)
    all_names = [x.var.name for x in tups]
    if len(tups) > 1:
        ctx.knob_manager.updateAxisParents(stage.op.name, fused.var.name, all_names)
    key = stage.op.name + '_' + fused.var.name
    ctx.scheduled_axes.append(key)
    if stage.op.name in ctx.compute_pos_names.keys():
        poses = ctx.compute_pos_names[stage.op.name]
        if ax.var.name in poses:
            idx = poses.index(ax.var.name)
            front = poses[:idx]
            back = poses[idx+1:]
            ctx.compute_pos_names[stage.op.name] = front + [fused.var.name] + back
    if update_dep_graph and len(tups) > 1:
        keys = [stage.op.name + "_" + ax.var.name for ax in tups]
        fused_key = stage.op.name + "_" + fused.var.name
        ctx.knob_manager.addFuse(keys, fused_key)
    return fused

def bind(ctx, stage, ax, threadtype):
    tx = te.thread_axis(threadtype)
    stage.bind(ax, tx)
    strs = "s[%s].bind(%s, te.thread_axis(\"%s\"))\n"%(
            getStageName(stage), getAxname(ax), threadtype
            )
    key = str(stage.op.name) + '_' + str(ax.var.name)
    ctx.addSchedDesc(strs)
    ctx.scheduled_axes.append(key)
    return tx

def unroll(ctx, stage, ax):
    stage.unroll(ax)
    strs = "s[%s].unroll(%s)\n"%(
            getStageName(stage), str(getAxname(ax)))
    key = str(stage.op.name) + '_' + str(ax.var.name)
    ctx.scheduled_axes.append(key)
    ctx.addSchedDesc(strs)
    ctx.unrolled_stages.append(stage.op.name)

def unrollPragma(ctx, stage, ax, num, explicit):
    stage.pragma(ax, "auto_unroll_max_step", num)
    stage.pragma(ax, "unroll_explicit", explicit)
    strs = "s[%s].pragma(%s, \"auto_unroll_max_step\", %d)\n"%(
            getStageName(stage), str(getAxname(ax)), num
            )
    if explicit:
        strs += "s[%s].pragma(%s, \"unroll_explicit\", True)\n"%(
                getStageName(stage), str(getAxname(ax))
                )
    else:
        strs += "s[%s].pragma(%s, \"unroll_explicit\", False)\n"%(
                getStageName(stage), str(getAxname(ax))
                )
    key = str(stage.op.name) + '_' + str(ax.var.name)
    ctx.scheduled_axes.append(key)
    ctx.addSchedDesc(strs)
    ctx.unrolled_stages.append(stage.op.name)
    ctx.axis_anotations[key] = "unroll"

def vectorize(ctx, stage, ax):
    stage.vectorize(ax)
    key = str(stage.op.name) + '_' + str(ax.var.name)
    ctx.scheduled_axes.append(key)
    strs = "s[%s].vectorize(%s)\n"%(getStageName(stage), str(getAxname(ax)))
    ctx.addSchedDesc(strs)
    ctx.vectorized_stages.append(stage.op.name)
    ctx.axis_anotations[key] = 'vectorize'

def double_buffer(ctx, stage):
    stage.double_buffer()
    strs = "s[%s].double_buffer()\n"%(getStageName(stage))
    ctx.addSchedDesc(strs)
    ctx.double_buffered_stages.append(stage.op.name)

def cache_read(ctx, stage_tensor, optype, out, sch):
    cached = sch.cache_read(stage_tensor, optype, [out])
    strs = "%s = s.cache_read(%s, %s, %s)\n"%(
            getTensorName(cached), getTensorName(stage_tensor), optype, out.name
            )
    ctx.addSchedDesc(strs)
    ctx.tensor_dict[cached.name] = cached
    return cached

def cache_write(ctx, stage_tensor, optype, sch):
    cached = sch.cache_write(stage_tensor, optype)
    strs = "%s = s.cache_write(%s, %s)\n"%(
            getTensorName(cached), getTensorName(stage_tensor), optype
            )
    ctx.addSchedDesc(strs)
    ctx.tensor_dict[cached.name] = cached
    return cached

def compute_at(ctx, stage, consumer_stage, consumer_ax):
    stage.compute_at(consumer_stage, consumer_ax)
    strs = "s[%s].compute_at(s[%s], %s)\n"%(
            getStageName(stage), getStageName(consumer_stage), getAxname(consumer_ax)
            )
    ctx.addSchedDesc(strs)

def compute_inline(ctx, stage):
    stage.compute_inline()
    strs = "s[%s].compute_line()\n"%(getStageName(stage))
    ctx.addSchedDesc(strs)
    ctx.no_schedule_stages.append(stage.op.name)

def pragma(ctx, stage, ax, strs):
    stage.pragma(ax, strs)
    strs = "s[%s].pragma(%s, %s)\n"%(getStageName(stage), getAxname(ax), strs)
    ctx.addSchedDesc(strs)

def set_scope(ctx, stage, scope):
    stage.set_scope(scope)
    strs = "s[%s].set_scope(%s)\n"%(getStageName(stage), str(scope))
    ctx.addSchedDesc(strs)


def tensorize(ctx, stage, ax, func_name, args):
    func_map = {
            'intrin_wmma_load_matrix_A': intrin_wmma_load_matrix_A,
            'intrin_wmma_load_matrix_W': intrin_wmma_load_matrix_W,
            'intrin_wmma_gemm': intrin_wmma_gemm,
            'intrin_wmma_store_matrix' : intrin_wmma_store_matrix,
            }
    
    func = func_map[func_name] 
    stage.tensorize(ax, func(*args))
    strs = "s[%s].tensorize(%s, %s(\n"%(getStageName(stage), getAxname(ax), func_name)
    for t in args:
        strs += str(t) + ', '
    strs += '\n))\n'
    ctx.addSchedDesc(strs)
    ctx.tensorized_stages.append(stage.op.name)
    # inner axes are all annotated with tensorize
    in_tensor_scope = False
    for tax in stage.leaf_iter_vars:
        if tax == ax:
            in_tensor_scope = True
        if in_tensor_scope:
            key = stage.op.name + '_' + tax.var.name
            ctx.axis_anotations[key] = 'tensorize'

def tensorize_x86(ctx, stage, ax, intrinsic):
    stage.tensorize(ax, intrinsic())
    strs = "s[%s].tensorize(%s, %s)\n"%(getStageName(stage), getAxname(ax), str(intrinsic))
    ctx.addSchedDesc(strs)
    ctx.tensorized_stages.append(stage.op.name)
    # inner axes are all annotated with tensorize
    in_tensor_scope = False
    for tax in stage.leaf_iter_vars:
        if tax == ax:
            in_tensor_scope = True
        if in_tensor_scope:
            key = stage.op.name + '_' + tax.var.name
            ctx.axis_anotations[key] = 'tensorize'

def tensorize_vta(ctx, stage, ax, intrinsic):
    stage.tensorize(ax, intrinsic)
    strs = "s[%s].tensorize(%s, %s)\n"%(getStageName(stage), getAxname(ax), str(intrinsic))
    ctx.addSchedDesc(strs)
    ctx.tensorized_stages.append(stage.op.name)
    # inner axes are all annotated with tensorize
    in_tensor_scope = False
    for tax in stage.leaf_iter_vars:
        if tax == ax:
            in_tensor_scope = True
        if in_tensor_scope:
            key = stage.op.name + '_' + tax.var.name
            ctx.axis_anotations[key] = 'tensorize'

def storage_align(ctx, stage, ax, a, b):
    stage.storage_align(ax, a, b)
    strs = "s[%s].storage_align(%s, %f, %f)\n"%(getStageName(stage), str(getAxname(ax)), a, b)
    ctx.addSchedDesc(strs)

def rfactor(ctx, tensor, ax, s):
    rf=s.rfactor(tensor, ax)
    strs = "s.rfactor(%s, %s)\n"%((tensor.op.name), str(getAxname(ax)))
    ctx.addSchedDesc(strs)
    return rf


