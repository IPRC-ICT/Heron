import os
import tvm
import shutil
import random
import numpy as np
import tvm.te as te
from tvm.topi import tag
from collections import deque

all_tags = ['blockIdx.x', \
            'vthread', \
            'threadIdx.x', \
            'threadIdx.y', \
            'Cthread', \
            'tensorize',\
            'vectorize', \
            'unroll', \
            'None'
            ]

def formattedList(l, num):
    strs = ''
    for x in l:
        strs += format(x, " ^" + str(num))
    return strs

class strBuffer:
    def __init__(self):
        self.val = ''
    
    def record(self, item):
        assert hasattr(item, '__str__')
        self.val += item.__str__() + '  \n'

def mkdir(name):
    if not os.path.exists(name):
        print("%s not exists, create one" %name)
        os.makedirs(name)
    else:
        print("%s exists, remove and create one" %name)
        shutil.rmtree(name)
        os.makedirs(name)

def mapNametoStage(sch, name):
    for stage in sch.stages:
        if stage.op.name == name:
            return stage
    raise ValueError("Not find in current schedule")

def mapNametoAxis(stage, name):
    i_vars = stage.leaf_iter_vars
    for v in i_vars:
        if v.var.name == name:
            return v
    # in stage.op.axis
    for ax in stage.op.axis:
        if ax.var.name == name:
            return ax
    # in stage.op.reduce_axis
    for ax in stage.op.reduce_axis:
        if ax.var.name == name:
            return ax
    print(i_vars)
    print(stage, name)
    raise ValueError("Not find in current schedule")

def getStageNamesOrdered(s):
    parents_map = {}
    zero_degrees = []
    all_names = [str(x.op.name) for x in s.stages]
  # assert 'compute' not in all_names
    for stage in s.stages:
        cur_stage_name = stage.op.name
        parent_names = []
        for x in s.stages:
            input_names = [t.name for t in x.op.input_tensors]
            if cur_stage_name in input_names:
                parent_names.append(x.op.name)
        parents_map[cur_stage_name] = parent_names
        if len(parent_names) == 0:
            zero_degrees.append(stage.op)
    ret = []
    while len(zero_degrees) > 0:
        cur = zero_degrees.pop()
        ret.append(cur.name)
        for x in cur.input_tensors:
            parents_map[x.name].remove(cur.name)
            if len(parents_map[x.name]) == 0:
                zero_degrees.append(x.op)
    return ret

def findStmtBufferSizes(stmt):
    size_map = {}
    def travel_func(stmt_exp):
        if isinstance(stmt_exp, tvm.tir.stmt.Allocate):
            var_name = stmt_exp.buffer_var.name
            var_shape = stmt_exp.extents
            var_size = 1
            for x in var_shape:
                var_size *= x.value
            size_map[var_name] = var_size
    tvm.tir.stmt_functor.post_order_visit(stmt, travel_func)
    return size_map

def getDefSplitCandidates(s, stage_name, ax_name, knob_manager, extent = 4096):
    key = stage_name + "_" + ax_name
    root_keys = knob_manager.get_axis_roots(key)
    # Only split non-fused axes
    assert len(root_keys) == 1
    stage = mapNametoStage(s, stage_name)
    root_ax_name = root_keys[0].split(stage_name + '_')[-1]
    root_ax = mapNametoAxis(stage, root_ax_name)
    dom_min, dom_extent = root_ax.dom.min, root_ax.dom.extent
    root_length = (dom_extent - dom_min).value
    res = []
    max_val = min(extent + 1, root_length + 1)
    for i in range(1, max_val):
        if root_length % i == 0:
            res.append(i)
    return res

def assert_print(bool_stmt, false_str=""):
    if not bool_stmt:
        raise AssertionError(false_str)


def getConsumers(s, stage_name):
    stage = mapNametoStage(s, stage_name)
    res = []
    for stage in s.stages:
        input_tensors = stage.op.input_tensors
        input_names = [x.name for x in input_tensors]
        if stage_name in input_names:
            res.append(stage.op.name)
    return res

def getProducers(s, stage_name):
    stage = mapNametoStage(s, stage_name)
    tensors = stage.op.input_tensors
    return [x.name for x in tensors]

def hasFusibleConsumer(s, stage_name):
    # Whether is the only producer of its consumer and only have 1 consummer
    consumers = getConsumers(s, stage_name)
    # 1 consumer
    if len(consumers) != 1:
        return None
    consumer = consumers[0]
    producers = getProducers(s, consumer)
    # 1 producer
    if len(producers) != 1:
        return None
    return consumer

def getStageName(stage):
    name = str(stage.op.name)
    return name.replace('.', '_')

def getAxname(ax):
    name = str(ax.var.name)
    name = name.replace('.', '_')
    name = name.replace("inner", 'i')
    name = name.replace("outer", 'o')
    name = name.replace("fused", 'f')
    return name

def getTensorName(tensor):
    name = str(tensor.name)
    return name.replace('.', '_')

def printAxes(s, stage_name, ctx):
    stage = mapNametoStage(s, stage_name)
    for ax in stage.leaf_iter_vars:
        key = stage_name  + "_" + ax.var.name
        if key in ctx.knob_manager.solver.vals.keys() and  ctx.knob_manager.get_val(key) != None:
            ctx.addSchedDesc("\n# Var %s length %d"%(getAxname(ax), ctx.knob_manager.get_val(key)))
        else:
            ctx.addSchedDesc("\n# Var %s"%(getAxname(ax)))

def genTCShape(ctx):
    # wmma_m, wmma_k, wmma_n
    wmma_m = ctx.knob_manager.get_val('wmma_m')
    wmma_k = ctx.knob_manager.get_val('wmma_k')
    wmma_n = ctx.knob_manager.get_val('wmma_n')
    shape = (wmma_m, wmma_n, wmma_k)
    return shape

def genTCLoadAParams(ctx, tc_shape):
    wmma_m, wmma_n, wmma_k = tc_shape
    name, (idx_1, idx_2), layout, dtype = ctx.tensorize_info['loadA']
    assert idx_2 > idx_1
    shape = [1] * (idx_2 - idx_1 + 1)
    if layout == "row_major":
        shape[0] = wmma_m; shape[-1] = wmma_k
        wmma_m_idx = idx_1; wmma_k_idx = idx_2
        tensorize_ax_idx = wmma_m_idx
        s_strides = []; l_strides = []
        for i in range(len(shape) - 1):
            s_strides.append(te.var('sa_k%d'%i))
            l_strides.append(te.var('la_k%d'%i))
        s_strides.append(1); l_strides.append(1)
    elif layout == "col_major":
        shape[0] = wmma_k; shape[-1] = wmma_m
        wmma_m_idx = idx_2; wmma_k_idx = idx_1
        tensorize_ax_idx = wmma_k_idx
        s_strides = []; l_strides = []
        for i in range(len(shape) - 1):
            s_strides.append(te.var('sa_m%d'%i))
            l_strides.append(te.var('la_m%d'%i))
        s_strides.append(1); l_strides.append(1)
    else:
        raise ValueError("Unsupportted layout %s"%layout)
    return s_strides, l_strides, shape, tensorize_ax_idx, wmma_m_idx, wmma_k_idx, dtype

def genTCLoadBParams(ctx, tc_shape):
    wmma_m, wmma_n, wmma_k = tc_shape
    name, (idx_1, idx_2), layout, dtype = ctx.tensorize_info['loadB']
    assert idx_2 > idx_1
    shape = [1] * (idx_2 - idx_1 + 1)
    if layout == "row_major":
        shape[0] = wmma_k; shape[-1] = wmma_n
        wmma_k_idx = idx_1; wmma_n_idx = idx_2
        tensorize_ax_idx = wmma_k_idx
        s_strides = []; l_strides = []
        for i in range(len(shape) - 1):
            s_strides.append(te.var('sb_n%d'%i))
            l_strides.append(te.var('lb_n%d'%i))
        s_strides.append(1); l_strides.append(1)
    elif layout == "col_major":
        shape[0] = wmma_n; shape[-1] = wmma_k
        wmma_k_idx = idx_2; wmma_n_idx = idx_1
        tensorize_ax_idx = wmma_n_idx
        s_strides = []; l_strides = []
        for i in range(len(shape) - 1):
            s_strides.append(te.var('sb_k%d'%i))
            l_strides.append(te.var('lb_k%d'%i))
        s_strides.append(1); l_strides.append(1)
    else:
        raise ValueError("Unsupportted layout %s"%layout)
    return s_strides, l_strides, shape, tensorize_ax_idx, wmma_k_idx, wmma_n_idx, dtype

def genTCStoreParams(ctx, tc_shape):
    wmma_m, wmma_n, wmma_k = tc_shape
    (wmma_m_idx, wmma_n_idx), dtype = ctx.tensorize_info['store']
    shape = [1] * (wmma_n_idx - wmma_m_idx + 1)
    shape[0] = wmma_m; shape[-1] = wmma_n
    tensorize_ax_idx = wmma_m_idx
    s_strides = []; l_strides = []
    for i in range(len(shape) - 1):
        l_strides.append(te.var('lc_n%d'%i))
        s_strides.append(te.var('sc_n%d'%i))
    s_strides.append(1); l_strides.append(1)
    return l_strides, s_strides, shape, tensorize_ax_idx, wmma_m_idx, wmma_n_idx, dtype

def getRootAttached(ctx, attached_name, producer):
    key = attached_name 
    pos_key = ctx.compute_poses[producer][1]
    while key in ctx.compute_poses.keys():
        pos_key = ctx.compute_poses[key][1]
        key = ctx.compute_poses[key][0]
    return key, pos_key

def get_divisable(num):
    res = []
    for i in range(1, num + 1):
        if num % i == 0:
            res.append(i)
    return res

def traverse_inline(s, final_op, ctx):
    visited = set()
    def _traverse(op):
        if op in visited:
            return
        visited.add(op)
        if tag.is_injective(op.tag):
            if op not in s.outputs:
                ctx.addSched('compute_inline', op.name, s)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.te.ComputeOp):
                    _traverse(tensor.op)
    _traverse(final_op)

def Code(point):
    assert isinstance(point, list)
    code = ''
    for x in point:
        code += str(x) + '_'
    return code

def DeCode(code):
    assert isinstance(code, str)
    nums = code.split('_')[:-1]
    return [int(x) for x in nums]


def anaCostModel(model, key_list):
    bst = model.bst
    if bst == None:
        random.shuffle(key_list)
        return key_list
    score_map = bst.get_score(importance_type = 'weight')
    score_list = [(key, score_map[key]) for key in score_map.keys()]
   #score_list_sorted = sorted(score_list, key = lambda x: x[1])[-30:]
    score_list_sorted = sorted(score_list, key = lambda x: -x[1])
    selected_idxs = [int(x[0].split('f')[1]) for x in score_list_sorted]
    selected_keys = [key_list[x] for x in selected_idxs]
    return selected_keys
    

def removedot(name):
    if isinstance(name, str):
        return name.replace('.', '_')
    return name


