import tvm
from tvm import te, auto_scheduler
from tvm import topi
from tvm.topi.x86.tensor_intrin import *
from tvm.topi.utils import get_const_tuple
import vta

vta_env = vta.get_env()

@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x

def heron_dense(ctx, M, K, N, in_dtype, out_dtype):
    a_shape = (M // vta_env.BATCH, K // vta_env.BLOCK_IN, vta_env.BATCH, vta_env.BLOCK_IN)
    b_shape = (N // vta_env.BLOCK_OUT, K // vta_env.BLOCK_IN, vta_env.BLOCK_OUT, vta_env.BLOCK_IN)
    o_shape = (M // vta_env.BATCH, N // vta_env.BLOCK_OUT, vta_env.BATCH, vta_env.BLOCK_OUT)
    A = te.placeholder(a_shape, name="A", dtype = vta_env.inp_dtype)
    B = te.placeholder(b_shape, name="B", dtype = vta_env.wgt_dtype)

    k_o = te.reduce_axis((0, a_shape[1]), name="k_o")
    k_i = te.reduce_axis((0, a_shape[3]), name="k_i")
    out = te.compute(
        o_shape,
        lambda b_o, c_o, b_i, c_i: te.sum(
            A[b_o, k_o, b_i, k_i].astype("int32")
            * B[c_o, k_o, c_i, k_i].astype("int32"),
            axis=[k_o, k_i],
        ),
        name = 'dense',
    )

    out = topi.right_shift(out, 8)
    out = my_clip(out, 0, 127)
    out = topi.cast(out, "int8")
    # pass information of tensorize strategy to optimizer
    info = {
            'name' : 'vta',
            'com' : ('dense', (2, 3, 1)),
            'intrin': [vta_env.gemm, vta_env.BATCH, vta_env.BLOCK_OUT, vta_env.BLOCK_IN]
            }
    ctx.set_info(info)
    axis_map = {
            'dense' : ['s_0', 's_1', 's_2', 's_3'],
            }
    ctx.set_axis_map(axis_map)
    return [out]
