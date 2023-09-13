import tvm
import tvm.autotvm as autotvm
import tvm.te as te
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import simplify
from tvm.topi.nn.pad import pad
from tvm.topi import tag

def heron_conv2d_nchw_tensorcore(ctx, N, H, W, CI, \
                                     CO, KH, KW,\
                                     stride, padding, dilation, in_dtype, out_dtype):
    """Compute declaration for tensorcore"""
    assert dilation == 1
    assert isinstance(stride, int) or len(stride) == 2

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    a_shape = (N, CI, H, W)
    w_shape = (CO, CI, KH, KW)
    Input = te.placeholder(a_shape, name="input", dtype = in_dtype)
    Filter = te.placeholder(w_shape, name="filter", dtype = in_dtype)

    # compute the output shape
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (KH, KW)
    )

    # For first convs
    if CI == 3:
        assert KH == 3 or KH == 7
        assert KW == 3 or KW == 7
        KH = KH + 1
        KW = KW + 1
        # Pad filter
        Filter = te.compute(
                [CO, CI, KH, KW],
                lambda co, ci, kh, kw: tvm.tir.if_then_else(
                    tvm.tir.all(kh < KH - 1, kw < KW - 1),
                    Filter[co, ci, kh, kw],
                    tvm.tir.const(0.0, Filter.dtype)
                    ),
                name = "FilterPad",
                tag = "injective,Filterpad"
                )
        pad_down = pad_down + 1
        pad_right = pad_right + 1
    else:
        assert CI % 16 == 0
    H_O = (H - KH + pad_top + pad_down) // stride_h + 1
    W_O = (W - KW + pad_left + pad_right) // stride_w + 1
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")

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
                i%(H_O*W_O)//W_O*stride_h+j%(KH*KW)//KW,
                i%W_O*stride_w+j%KW
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

    # Use stride to restrict the choice of vectorization
    # e.g., 147 x 147 can not be vectorized
    output = te.compute(
        [N, CO, H_O, W_O],
        lambda n, c, p, q:
            C[c, n * (H_O*W_O) + p * W_O + q],
        name="output",
        tag="default, stride:%d"%(H_O * W_O)
    )
  # output = C

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

def heron_conv2d_nhwc_tensorcore(ctx, N, H, W, CI, \
                                     CO, KH, KW,\
                                     stride, padding, dilation, in_dtype, out_dtype):
    """Compute declaration for tensorcore"""
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation = dilation = dilation
    else:
        dilation, dilation = dilation

    a_shape = (N, H, W, CI)
    w_shape = (KH, KW, CI, CO)
    Input = te.placeholder(a_shape, name="input", dtype = in_dtype)
    Filter = te.placeholder(w_shape, name="filter", dtype = in_dtype)
    assert (
        (N % 16 == 0 and CI % 16 == 0 and CO % 16 == 0)
        or (N % 8 == 0 and CI % 16 == 0 and CO % 32 == 0)
        or (N % 32 == 0 and CI % 16 == 0 and CO % 8 == 0)
    ), (
        "The shape of (N, CI, CO) "
        "must be multiple of (16, 16, 16) or (32, 16, 8) or (8, 16, 32) for now"
    )

    # compute the output shape
    dilated_KH = (KH - 1) * dilation + 1
    dilated_KW = (KW - 1) * dilation + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_KH, dilated_KW)
    )
    CO = CO
    H_O = simplify((H - dilated_KH + pad_top + pad_down) // stride_h + 1)
    W_O = simplify((W - dilated_KW + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    rc = te.reduce_axis((0, CI), name="rc")
    ry = te.reduce_axis((0, KH), name="ry")
    rx = te.reduce_axis((0, KW), name="rx")
    Output = te.compute(
        (N, H_O, W_O, CO),
        lambda nn, yy, xx, ff: te.sum(
            PaddedInput[
                nn,
                yy * stride_h + ry * dilation,
                xx * stride_w + rx * dilation,
                rc
            ].astype(out_dtype)
            * Filter[ry, rx, rc, ff].astype(out_dtype),
            axis=[ry, rx, rc],
        ),
        name="out",
        tag="conv2d_nhwc_tensorcore",
    )

    # pass information of tensorcore strategy to optimizer
    def tensorCoreCompute(cl_shape, AL_shape, WL_shape, in_dtype, out_dtype):
        wmma_k = 16
        k_gemm = te.reduce_axis((0, wmma_k), name="k")
        AL_gemm = te.placeholder(AL_shape, name="A", dtype=in_dtype)
        WL_gemm = te.placeholder(WL_shape, name="B", dtype=in_dtype)
        cl_compute = te.compute(cl_shape, 
                lambda ii, t0, t1, jj: te.sum(
                    AL_gemm[ii, t0, t1, k_gemm].astype(out_dtype) *
                    WL_gemm[k_gemm, jj].astype(out_dtype),
                    axis = k_gemm,
                    ),
                    name = "C"
                )
        return AL_gemm, WL_gemm, cl_compute
    info = {
            'name' : 'tensorcore',
            'stage_name' : Output.op.name,
            'loadA' : (PaddedInput.name, (0, 3), 'row_major', in_dtype),
            'loadB' : (Filter.name, (2, 3), 'row_major', in_dtype),
            'com' : ((0, 2, 3), out_dtype),
            'store' : ((0, 3), out_dtype),
            'compute_func' : tensorCoreCompute
            }
    ctx.set_info(info)
    axis_map = {
            'PaddedInput.shared' : ['s_0', 's_1', 's_2', 'r_2'],
            'PaddedInput.shared.wmma.matrix_a' : ['s_0', 's_1', 's_2', 'r_2'],
            'filter.shared' : ['r_0', 'r_1', 'r_2', 's_3'],
            'filter.shared.wmma.matrix_b' : ['r_0', 'r_1', 'r_2', 's_3'],
            'out.wmma.accumulator.shared' : ['s_0', 's_1', 's_2', 's_3'],
            'out.wmma.accumulator' : ['s_0', 's_1', 's_2', 's_3'],
            }
    ctx.set_axis_map(axis_map)
    return [Output]


