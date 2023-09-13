import tvm
import time
import tvm.autotvm as autotvm
import tvm.te as te
from tvm.topi.nn.utils import get_pad_tuple3d
from tvm.topi.utils import simplify, get_const_tuple
from tvm.topi.nn.pad import pad
from tvm.topi import tag

def heron_conv3d_ncdhw_tensorcore(ctx, N, D, H, W, CI, \
                                     CO, KD, KH, KW,\
                                     stride, padding,\
                                     dilation, in_dtype, out_dtype):
    """Compute declaration for tensorcore"""
    assert dilation == 1
    assert isinstance(stride, int) or len(stride) == 3
    if isinstance(stride, int):
        stride_d = stride_h = stride_w= stride
    else:
        stride_d, stride_h, stride_w = stride


    a_shape = (N, CI, D, H, W)
    w_shape = (CO, CI, KD, KH, KW)
    Input = te.placeholder(a_shape, name="input", dtype = in_dtype)
    Filter = te.placeholder(w_shape, name="filter", dtype = in_dtype)

    # compute the output shape
    pad_front, pad_top, pad_left, pad_back, pad_down, pad_right = get_pad_tuple3d(
        padding, (KD, KH, KW)
    )
    out_channel = CO
    D_O = simplify((D - KD + pad_front + pad_back) // stride_d + 1)
    H_O = simplify((H - KH + pad_top + pad_down) // stride_h + 1)
    W_O = simplify((W - KW + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, 0, pad_front, pad_top, pad_left]
    pad_after = [0, 0, pad_back, pad_down, pad_right]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")

    # Im2col
    MM = N * D_O * H_O * W_O
    MN = CO
    MK = CI * KD * KH * KW
    A = te.compute(
        [MM, MK],
        lambda i, j:
            PaddedInput[
                i//(D_O*H_O*W_O),
                j//(KD*KH*KW),
                i%(D_O*H_O*W_O)//(H_O*W_O)*stride_d+j%(KD*KH*KW)//(KH*KW),
                i%(H_O*W_O)//W_O*stride_h+j%(KH*KW)//KW,                
                i%W_O*stride_w+j%KW
            ],
        name="A",
        tag="injective,A"
    )
    B = te.compute(
        [MN, MK],
        lambda i, j:
            Filter[i, j//(KD*KH*KW), j%(KD*KH*KW)//(KH*KW), j%(KH*KW)//KW, j%KW],
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
        [N, CO, D_O, H_O, W_O],
        lambda n, c, d, p, q:
            C[c, n * (D_O*H_O*W_O) + d * H_O * W_O + p * W_O + q],
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

def heron_conv3d_ndhwc_tensorcore(ctx, N, D, H, W, CI,
                                    CO, KD, KH, KW,
                                    stride,
                                    padding,
                                    dilation,
                                    in_dtype,
                                    out_dtype):
    """Compute declaration for conv3d tensorcore function"""
    assert isinstance(stride, int) or len(stride) == 3
    assert isinstance(dilation, int) or len(dilation) == 3

    if isinstance(stride, int):
        stride_d = stride_h = stride_w= stride
    else:
        stride_d, stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation = dilation = dilation = dilation
    else:
        dilation, dilation, dilation = dilation

    a_shape = (N, D, H, W, CI)
    w_shape = (KD, KH, KW, CI, CO)
    Input = te.placeholder(a_shape, name="input", dtype = in_dtype)
    Filter = te.placeholder(w_shape, name="filter", dtype = in_dtype)
    assert(
        (N % 16 == 0 and CI % 16 == 0 and CO % 16 == 0)
        or (N % 8 == 0 and CI % 16 == 0 and CO % 32 == 0)
        or (N % 32 == 0 and CI % 16 == 0 and CO % 8 == 0)
    ), (
        "The shape of (N, CI, CO) "
        "must be multiple of (16, 16, 16) or (32, 16, 8) or (8, 16, 32) for now"
    )

    # compute the output shape
    dilated_KD = (KD - 1) * dilation + 1
    dilated_KH = (KH - 1) * dilation + 1
    dilated_KW = (KW - 1) * dilation + 1
    pad_front, pad_top, pad_left, pad_back, pad_down, pad_right = get_pad_tuple3d(
        padding, (dilated_KD, dilated_KH, dilated_KW)
    )
    out_channel = CO
    D_O = simplify((D - dilated_KD + pad_front + pad_back) // stride_d + 1)
    H_O = simplify((H - dilated_KH + pad_top + pad_down) // stride_h + 1)
    W_O = simplify((W - dilated_KW + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_front, pad_top, pad_left, 0]
    pad_after = [0, pad_back, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    rc = te.reduce_axis((0, CI), name="rc")
    rz = te.reduce_axis((0, KD), name="rz")
    ry = te.reduce_axis((0, KH), name="ry")
    rx = te.reduce_axis((0, KW), name="rx")
    # convert data type of input feature maps and weights
    # TODO: add checking here, datatype casting may cause precision loss
    TransPaddedInput = te.compute(
        PaddedInput.shape, lambda n, d, h, w, c: PaddedInput[n, d, h, w, c].astype(in_dtype),
        name = "TransPaddedInput",
        tag = tag.INJECTIVE+",transpad"
    )
    TransFilter = te.compute(
        Filter.shape, lambda d, h, w, i, o: Filter[d, h, w, i, o].astype(in_dtype),
        name = "TransFilter",
        tag = tag.INJECTIVE+",transfilter"
    )
    Output = te.compute(
        (N, D_O, H_O, W_O, out_channel),
        lambda nn, zz, yy, xx, ff: te.sum(
            TransPaddedInput[
                nn,
                zz * stride_d + rz * dilation,
                yy * stride_h + ry * dilation,
                xx * stride_w + rx * dilation,
                rc,
            ].astype(out_dtype)
            * TransFilter[rz, ry, rx, rc, ff].astype(out_dtype),
            axis=[rz, ry, rx, rc],
        ),
        name="out",
        tag="conv3d_ndhwc_tensorcore",
    )

    # pass information of tensorcore strategy to optimizer
    def tensorCoreCompute(cl_shape, AL_shape, WL_shape, in_dtype, out_dtype):
        wmma_k = 16
        k_gemm = te.reduce_axis((0, wmma_k), name="k")
        AL_gemm = te.placeholder(AL_shape, name="A", dtype=in_dtype)
        WL_gemm = te.placeholder(WL_shape, name="B", dtype=in_dtype)
        cl_compute = te.compute(cl_shape, 
                lambda ii, t0, t1, t2, jj: te.sum(
                    AL_gemm[ii, t0, t1, t2, k_gemm].astype(out_dtype) *
                    WL_gemm[k_gemm, jj].astype(out_dtype),
                    axis = k_gemm,
                    ),
                    name = "C"
                )
        return AL_gemm, WL_gemm, cl_compute
    info = {
            'name' : 'tensorcore',
            'stage_name' : Output.op.name,
            'loadA' : (TransPaddedInput.name, (0, 4), 'row_major', in_dtype),
            'loadB' : (TransFilter.name, (3, 4), 'row_major', in_dtype),
            'com' : ((0, 3, 4), out_dtype),
            'store' : ((0, 4), out_dtype),
            'compute_func' : tensorCoreCompute
            }
    ctx.set_info(info)
    axis_map = {
            'TransPaddedInput.shared' : ['s_0', 's_1', 's_2', 's_3', 'r_3'],
            'TransPaddedInput.shared.wmma.matrix_a' : ['s_0', 's_1', 's_2', 's_3', 'r_3'],
            'TransFilter.shared' : ['r_0', 'r_1', 'r_2', 'r_3', 's_4'],
            'TransFilter.shared.wmma.matrix_b' : ['r_0', 'r_1', 'r_2', 'r_3', 's_4'],
            'out.wmma.accumulator.shared' : ['s_0', 's_1', 's_2', 's_3', 's_4'],
            'out.wmma.accumulator' : ['s_0', 's_1', 's_2', 's_3', 's_4'],
            }
    ctx.set_axis_map(axis_map)
    return [Output]


