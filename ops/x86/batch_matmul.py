import tvm
from tvm import te, auto_scheduler
from tvm.topi.x86.tensor_intrin import *
from tvm.topi.utils import get_const_tuple

def heron_batch_matmul(ctx, batch, M, K, N, in_dtype, out_dtype):
    assert batch > 1
    a_shape = (batch, M, K)
    b_shape = (batch, N, K)
    o_shape = (batch, M, N)
    A = te.placeholder(a_shape, name="A", dtype = 'uint8')
    B = te.placeholder(b_shape, name="B", dtype = 'int8')
    k = te.reduce_axis((0, K), name="k")

    out = te.compute(
        o_shape,
        lambda b, i, j: te.sum(A[b, i, k].astype('int32') * B[b, j, k].astype('int32'), axis=k),
        name = 'batch_matmul',
        tag="batch_matmul",
    )

    # pass information of tensorize strategy to optimizer
    info = {
            'name' : 'x86_dot',
            'stage_orgination' : [('batch_matmul', 'addCacheWriteGlobal')],
            'com' : ('batch_matmul.global', (2, 0)),
            'intrin': [dot_16x1x16_uint8_int8_int32_cascadelake, 16, 4]
            }
    ctx.set_info(info)
    axis_map = {
            'batch_matmul.global' : ['s_0', 's_1', 's_2'],
            }
    ctx.set_axis_map(axis_map)
    return [out]

