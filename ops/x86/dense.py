import tvm
from tvm import te, auto_scheduler
from tvm.topi.x86.tensor_intrin import *
from tvm.topi.utils import get_const_tuple

def heron_dense(ctx, M, K, N, in_dtype, out_dtype):
    a_shape = (M, K)
    b_shape = (N, K)
    o_shape = (M, N)
    A = te.placeholder(a_shape, name="A", dtype = 'uint8')
    B = te.placeholder(b_shape, name="B", dtype = 'int8')
    k = te.reduce_axis((0, K), name="k")

    out = te.compute(
        o_shape,
        lambda i, j: te.sum(A[i, k].astype('int32') * B[j, k].astype('int32'), axis=k),
        name = 'dense',
    )

    # pass information of tensorize strategy to optimizer
    info = {
            'name' : 'x86_dot',
            'stage_orgination' : [('dense', 'addCacheWriteGlobal')],
            'com' : ('dense.global', (1, 0)),
            'intrin': [dot_16x1x16_uint8_int8_int32_cascadelake, 16, 4]
            }
    ctx.set_info(info)
    axis_map = {
            'dense.global' : ['s_0', 's_1'],
            }
    ctx.set_axis_map(axis_map)
    return [out]
