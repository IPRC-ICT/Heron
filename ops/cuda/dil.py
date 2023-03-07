import tvm.autotvm as autotvm
import tvm.te as te
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import simplify
from tvm.topi.nn.pad import pad
from tvm.topi import tag

def heron_dil_tensorcore(ctx, N, H, W, CI, \
                                     CO, KH, KW,\
                                     stride, padding, dilation, in_dtype, out_dtype):
    """Compute declaration for tensorcore"""
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(stride, int):
        stride = stride = stride
    else:
        stride, stride = stride

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
    H_O = simplify((H - dilated_KH + pad_top + pad_down) // stride + 1)
    W_O = simplify((W - dilated_KW + pad_left + pad_right) // stride + 1)
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
                yy * stride + ry * dilation,
                xx * stride + rx * dilation,
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


