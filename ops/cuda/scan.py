import tvm
from tvm import te, auto_scheduler
from tvm.topi.utils import get_const_tuple
def heron_scan(ctx, M, N, K, in_dtype, out_dtype):
    A = tvm.te.placeholder([M, K], dtype=in_dtype, name="A")
    B = tvm.te.compute(
        [N, K],
        lambda i, j: tvm.tir.if_then_else(
            i <= j, tvm.tir.const(0.0, in_dtype), tvm.tir.const(1.0, "float16")
        ),
        name="B",
        tag="injective,B"
    )

    k = te.reduce_axis((0, K), name="k")

    out = te.compute(
        [M, N],
        lambda i, j: te.sum(A[i, k].astype(out_dtype) * B[j, k].astype(out_dtype), axis=k),
        name = 'dense',
        tag="matmul",
    )

    def tensorCoreCompute(cl_shape, AL_shape, WL_shape, in_dtype, out_dtype):
        wmma_k = 16
        k_gemm = te.reduce_axis((0, wmma_k), name="wmma_k")
        AL_gemm = te.placeholder(AL_shape, name="wmma_A", dtype=in_dtype)
        WL_gemm = te.placeholder(WL_shape, name="wmma_B", dtype=in_dtype)
        cl_compute = te.compute(
            cl_shape,
            lambda ii, jj: te.sum(
                AL_gemm[ii, k_gemm].astype(out_dtype) * WL_gemm[jj, k_gemm].astype(out_dtype),
                axis=k_gemm,
            ),
            name="wmma_C",
        )
        return AL_gemm, WL_gemm, cl_compute

    # pass information of tensorcore strategy to optimizer
    info = {
            'name' : 'tensorcore',
            'stage_name' : out.op.name,
            'loadA' : (A.name, (0, 1), 'row_major', in_dtype),
            'loadB' : (B.name, (0, 1), 'col_major', in_dtype),
            'com' : ((0, 0, 1), out_dtype),
            'store' : ((0, 1), out_dtype),
            'compute_func' : tensorCoreCompute
            }
    ctx.set_info(info)
    axis_map = {
            'A.shared' : ['s_0', 'r_0'],
            'A.shared.wmma.matrix_a' : ['s_0', 'r_0'],
            'B.shared' : ['s_1', 'r_0'],
            'B.shared.wmma_matrix_b' : ['s_1', 'r_0'],
            'dense.wmma.accumulator.shared' : ['s_0', 's_1'],
            'dense.wmma.accumulator' : ['s_0', 's_1']
            }
    ctx.set_axis_map(axis_map)
    return [out]
