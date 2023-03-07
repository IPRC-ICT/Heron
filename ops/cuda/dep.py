import tvm
import tvm.autotvm as autotvm
import tvm.te as te
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import simplify
from tvm.topi.nn.pad import pad
from tvm.topi import tag

def heron_depthwise_conv2d_tensorcore(ctx, N, H, W, C, \
                                     K, KH, KW,\
                                     stride, padding, dilation, in_dtype, out_dtype):
    """compute declaration for tensorcore"""
    assert isinstance(stride, int)
    assert dilation == 1
    assert K % C == 0
    assert KH == 3 and KW == 3

    R_align = 4
    S_align = 4
    a_shape = (N, C, H, W)
    w_shape = (K, KH, KW)
    pH = H + 2 * padding + R_align - KH
    pW = W + 2 * padding + S_align - KW
    Input = te.placeholder(a_shape, name="input", dtype = in_dtype)
    Filter = te.placeholder(w_shape, name="filter", dtype = in_dtype)

    PaddedInput = te.compute(
        [N, C, pH, pW],
        lambda n, c, h, w: tvm.tir.if_then_else(
            tvm.tir.all(h >= padding, h - padding < H,
                        w >= padding, w - padding < W),
            Input[n, c, h - padding, w - padding],
            tvm.tir.const(0.0, Input.dtype),
        ),
        name="Apad",
        tag = "injective,Apad"
    )
    H_O = (H - KH + 2*padding) // stride + 1
    W_O = (W - KW + 2*padding) // stride + 1

    # Im2col
    MB = C
    MM = N * H_O * W_O
    MN_raw = (K // C)
    MN = (MN_raw + 7) // 8 * 8
    MK =  R_align * S_align
    A = te.compute(
        [MB, MM, MK],
        lambda b, i, j:
            PaddedInput[
                i//(H_O*W_O),
                b,
                i%(H_O*W_O)//W_O*stride+j%(R_align*S_align)//S_align,
                i%W_O*stride+j%S_align
            ],
        name="A",
        tag="injective,A"
    )
    B = tvm.te.compute(
        [MB, MN, MK],
        lambda k_o, k_i, rs: tvm.tir.if_then_else(
            tvm.tir.all(rs // S_align >=0, rs // S_align < 3,
                        rs % S_align >= 0, rs % S_align < 3,
                        k_i >= 0, k_i < MN_raw),
                Filter[k_o * (K // C) + k_i, rs // S_align, rs % S_align],
                tvm.tir.const(0.0, Filter.dtype),
            ),
            name="B",
            tag="injective,B"
    )

    k = te.reduce_axis((0, MK), name="k")
    CM = te.compute(
        [MB, MN, MM],
        lambda b, i, j: te.sum(B[b, i, k].astype(out_dtype) * \
                            A[b, j, k].astype(out_dtype), axis=k),
        name = 'C'
    )

    output = te.compute(
        [N, C, MN, H_O, W_O],
        lambda n, c1, c2, p, q:
            CM[c1, c2, n * (H_O*W_O) + p * W_O + q],
        name="output",
        tag="default"
    )

    # pass information of tensorcore strategy to optimizer
    def tensorcorecompute(cl_shape, AL_shape, WL_shape, in_dtype, out_dtype):
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
            'stage_name' : CM.op.name,
            'loadA' : (B.name, (1, 2), 'row_major', in_dtype),
            'loadB' : (A.name, (1, 2), 'col_major', in_dtype),
            'com' : ((1, 0, 2), out_dtype),
            'store' : ((1, 2), out_dtype),
            'compute_func' : tensorcorecompute
            }
    ctx.set_info(info)
    axis_map = {
            'B.shared' : ['s_0', 's_1', 'r_0'],
            'B.shared.wmma.matrix_a' : ['s_0', 's_1', 'r_0'],
            'A.shared' : ['s_1', 's_2', 'r_0'],
            'A.shared.wmma.matrix_b' : ['s_1', 's_2', 'r_0'],
            'C.wmma.accumulator.shared' : ['s_0', 's_1', 's_2'],
            'C.wmma.accumulator' : ['s_0', 's_1', 's_2'],
            }
    ctx.set_axis_map(axis_map)
    return [output]

