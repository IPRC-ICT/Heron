import tvm
import time
import tvm.autotvm as autotvm
import tvm.te as te
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import simplify, get_const_tuple
from tvm.topi.nn.pad import pad
import tvm.topi.nn as nn
from tvm.topi import tag
def zero_expand2d(inputs, stride=1):
    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    assert isinstance(stride, tuple), "type(stride)={}".format(type(stride))
    assert len(stride) == 2

    expand_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel, height, width = inputs.shape
    out_height = (height - 1) * stride[0] + 1
    out_width = (width - 1) * stride[1] + 1
    return tvm.te.compute(
        (batch_size, in_channel, out_height, out_width),
        lambda b, c, h, w: tvm.te.if_then_else(
            tvm.te.all(h % stride[0] == 0, w % stride[1] == 0),
            inputs[b, c, h // stride[0], w // stride[1]],
            expand_zero,
        ),
        name = "expand2d",
        tag = "injective,expand2d"
    )

def zero_pad2d(inputs, padding=0):
    padding = (
        (padding, padding, padding, padding)
        if isinstance(padding, (int, tvm.tir.IntImm))
        else padding
    )
    assert isinstance(padding, tuple), "type(padding)={}".format(type(padding))
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert len(padding) == 4

    padding_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel, height, width = inputs.shape
    return tvm.te.compute(
        (batch_size, in_channel, height + padding[0] + padding[1], width + padding[2] + padding[3]),
        lambda b, c, h, w: tvm.te.if_then_else(
            tvm.te.all(
                h >= padding[0], h < height + padding[0], w >= padding[2], w < width + padding[2]
            ),
            inputs[b, c, h - padding[0], w - padding[2]],
            padding_zero,
        ),
        name="Padding",
        tag = "injective,pad"
    )


def heron_conv2d_nchw_transposed_tensorcore(ctx,
                                 N, H, W, CI,
                                 CO, KH, KW,
                                 stride,
                                 padding,
                                 dilation,
                                 outpadding,
                                 in_dtype,
                                 out_dtype):
    """Compute declaration for tensorcore"""
    assert isinstance(padding, int)
    assert isinstance(stride, int)
    assert isinstance(dilation, int)

    a_shape = (N, CI, H, W)
    w_shape = (CO, CI, KH, KW)
    Input = te.placeholder(a_shape, name="input", dtype = in_dtype)
    Filter = te.placeholder(w_shape, name="filter", dtype = in_dtype)

    # compute the output shape
    dilated_KH = (KH - 1) * dilation + 1
    dilated_KW = (KW - 1) * dilation + 1
    H_O = (H - 1) * stride - 2 * padding + dilated_KH + outpadding
    W_O = (W - 1) * stride - 2 * padding + dilated_KW + outpadding
    expanded = zero_expand2d(Input, stride)
    PaddedInput = zero_pad2d(
        expanded,
        padding=(
            dilated_KH - 1 - padding,
            dilated_KH - 1 - padding + outpadding,
            dilated_KW - 1 - padding,
            dilated_KW - 1 - padding + outpadding,
        )
    )

    # Im2col
    MM = N * H_O * W_O
    MN = CO
    MK = CI * KH * KW
    A = te.compute(
        [MM, MK],
        lambda i, j:
            PaddedInput[
                i//(H_O*W_O),
                j//(KH*KW),
                i%(H_O*W_O)//W_O*stride+j%(KH*KW)//KW,
                i%W_O*stride+j%KW
            ],
        name="A",
        tag="injective,A"
    )
    B = te.compute(
        [MN, MK],
        lambda i, j:
            Filter[i, j//(KH*KW), j%(KH*KW)//KW, j%KW],
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
        [N, CO, H_O, W_O],
        lambda n, c, p, q:
            C[c, n * (H_O*W_O) + p * W_O + q],
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



