import tvm
import time
import tvm.autotvm as autotvm
import tvm.te as te
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import simplify
from tvm.topi.nn.utils import get_pad_tuple1d
from tvm.topi.nn.pad import pad
from tvm.topi import tag

def heron_conv1d_ncw_tensorcore(ctx, N, W, CI, \
                                     CO, KW,\
                                     stride, padding, dilation, in_dtype, out_dtype):
    """Compute declaration for tensorcore"""
    assert isinstance(stride, int)
    # Use dilated im2col or use nwc layout
    assert dilation == 1

    a_shape = (N, CI, W)
    w_shape = (CO, CI, KW)
    Input = te.placeholder(a_shape, name="input", dtype = in_dtype)
    Filter = te.placeholder(w_shape, name="filter", dtype = in_dtype)

    # compute the output shape
    pad_left, pad_right = get_pad_tuple1d(padding, (KW,))
    W_O = simplify((W - KW + pad_left + pad_right) // stride + 1)

    # Apply padding
    pad_before = [0, 0, pad_left]
    pad_after = [0, 0, pad_right]
    PaddedInput = pad(Input, pad_before, pad_after, name="pad")

    # Im2col
    MM = N * W_O
    MN = CO
    MK = CI * KW
    A = te.compute(
        [MM, MK],
        lambda i, j:
            PaddedInput[
                i//W_O,
                j//KW,
                i%W_O*stride+j%KW
            ],
        name="A",
        tag="injective,A"
    )
    B = te.compute(
        [MN, MK],
        lambda i, j:
            Filter[i, j//KW, j%KW],
        name="B",
        tag="injective,B"
    )

    k = te.reduce_axis((0, MK), name="k")
    C = te.compute(
        [MN, MM],
        lambda i, j: te.sum(B[i, k].astype(out_dtype) * \
                            A[j, k].astype(out_dtype), axis=k),
        name = 'C'
    )

    output = te.compute(
        [N, CO, W_O],
        lambda n, c, q:
            C[c, n * W_O + q],
        name="output",
        tag="default"
    )

    # pass information of tensorcore strategy to optimizer
    def tensorCoreCompute(cl_shape, AL_shape, WL_shape, in_dtype, out_dtype):
        wmma_k = 16
        k_gemm = te.reduce_axis((0, wmma_k), name="k")
        AL_gemm = te.placeholder(AL_shape, name="A_gemm", dtype=in_dtype)
        WL_gemm = te.placeholder(WL_shape, name="B_gemm", dtype=in_dtype)
        cl_compute = te.compute(cl_shape, 
                lambda ii,  jj: te.sum(
                    AL_gemm[ii, k_gemm].astype(out_dtype) *
                    WL_gemm[jj, k_gemm].astype(out_dtype),
                    axis = k_gemm,
                    ),
                    name = "C_gemm"
                )
        return AL_gemm, WL_gemm, cl_compute
    info = {
            'name' : 'tensorcore',
            'stage_name' : C.op.name,
            'loadA' : (B.name, (0, 1), 'row_major', in_dtype),
            'loadB' : (A.name, (0, 1), 'col_major', in_dtype),
            'com' : ((0, 0, 1), out_dtype),
            'store' : ((0, 1), out_dtype),
            'compute_func' : tensorCoreCompute
            }
    ctx.set_info(info)
    axis_map = {
            'B.shared' : ['s_0', 'r_0'],
            'B.shared.wmma.matrix_a' : ['s_0', 'r_0'],
            'A.shared' : ['s_1', 'r_0'],
            'A.shared.wmma.matrix_b' : ['s_1', 'r_0'],
            'C.wmma.accumulator.shared' : ['s_0', 's_1'],
            'C.wmma.accumulator' : ['s_0', 's_1'],
            }
    ctx.set_axis_map(axis_map)
    return [output]


def heron_conv1d_nwc_tensorcore(ctx,
                                N,
                                W,
                                CI,
                                CO,
                                K,
                                stride,
                                padding,
                                dilation,
                                in_dtype,
                                out_dtype):
    assert isinstance(stride, int)
    assert isinstance(dilation, int)

    a_shape = (N, W, CI)
    w_shape = (K, CI, CO)
    Input = te.placeholder(a_shape, name="input", dtype = in_dtype)
    Filter = te.placeholder(w_shape, name="filter", dtype = in_dtype)

    # Compute the output shape
    dilated_K = (K - 1) * dilation + 1
    pad_left, pad_right = get_pad_tuple1d(padding, (dilated_K,))
    CO = simplify(CO)
    W_O = simplify((W - dilated_K + pad_left + pad_right) // stride + 1)

    # Apply padding
    pad_before = [0, pad_left, 0]
    pad_after = [0, pad_right, 0]
    temp = pad(Input, pad_before, pad_after, name="pad")

    # Compute graph
    rc = te.reduce_axis((0, CI), name="rc")
    rw = te.reduce_axis((0, K), name="rw")

    out = te.compute(
        (N, W_O, CO),
        lambda b, w, c: te.sum(
            temp[b, w * stride + rw * dilation, rc].astype(out_dtype)
            * Filter[rw, rc, c].astype(out_dtype),
            axis=[rw, rc],
        ),
        name = 'out',
        tag="conv1d_nwc",
    )
    # pass information of tensorcore strategy to optimizer
    def tensorCoreCompute(cl_shape, AL_shape, WL_shape, in_dtype, out_dtype):
        wmma_k = 16
        k_gemm = te.reduce_axis((0, wmma_k), name="k")
        AL_gemm = te.placeholder(AL_shape, name="A", dtype=in_dtype)
        WL_gemm = te.placeholder(WL_shape, name="B", dtype=in_dtype)
        cl_compute = te.compute(cl_shape, 
                lambda ii, t, jj: te.sum(
                    AL_gemm[ii, t, k_gemm].astype(out_dtype) *
                    WL_gemm[k_gemm, jj].astype(out_dtype),
                    axis = k_gemm,
                    ),
                    name = "C"
                )
        return AL_gemm, WL_gemm, cl_compute
    info = {
            'name' : 'tensorcore',
            'stage_name' : out.op.name,
            'loadA' : (temp.name, (0, 2), 'row_major', in_dtype),
            'loadB' : (Filter.name, (1, 2), 'row_major', in_dtype),
            'com' : ((0, 1, 2), out_dtype),
            'store' : ((0, 2), out_dtype),
            'compute_func' : tensorCoreCompute
            }
    ctx.set_info(info)
    axis_map = {
            'pad.shared' : ['s_0', 's_1', 'r_1'],
            'pad.shared.wmma.matrix_a' : ['s_0', 's_1', 'r_1'],
            'filter.shared' : ['r_0', 'r_1', 's_2'],
            'filter.shared.wmma.matrix_b' : ['r_0', 'r_1', 's_2'],
            'out.wmma.accumulator.shared' : ['s_0', 's_1', 's_2'],
            'out.wmma.accumulator' : ['s_0', 's_1', 's_2'],
            }
    ctx.set_axis_map(axis_map)
    return [out]
