import tvm
from tvm import te, auto_scheduler
from tvm.topi.utils import get_const_tuple

def heron_batch_matmul(ctx, batch, M, K, N, in_dtype, out_dtype):
    assert batch > 1
    a_shape = (batch, M, K)
    b_shape = (batch, N, K)
    o_shape = (batch, M, N)
    A = te.placeholder(a_shape, name="A", dtype = in_dtype)
    B = te.placeholder(b_shape, name="B", dtype = in_dtype)
    k = te.reduce_axis((0, K), name="k")

    out = te.compute(
        o_shape,
        lambda b, i, j: te.sum(A[b, i, k].astype(out_dtype) * B[b, j, k].astype(out_dtype), axis=k),
        name = 'batch_matmul',
        tag="batch_matmul",
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
            'loadA' : (A.name, (1, 2), 'row_major', 'float16'),
            'loadB' : (B.name, (1, 2), 'col_major', 'float16'),
            'com' : ((1, 0, 2), out_dtype),
            'store' : ((1, 2), out_dtype),
            'compute_func' : tensorCoreCompute
            }
    ctx.set_info(info)
    axis_map = {
            'A.shared' : ['s_0', 's_1', 'r_0'],
            'A.shared.wmma.matrix_a' : ['s_0', 's_1', 'r_0'],
            'B.shared' : ['s_0', 's_2', 'r_0'],
            'B.shared.wmma.matrix_b' : ['s_0', 's_2', 'r_0'],
            'batch_matmul.wmma.accumulator.shared' : ['s_0', 's_1', 's_2'],
            'batch_matmul.wmma.accumulator' : ['s_0', 's_1', 's_2']
            }
    ctx.set_axis_map(axis_map)
    return [out]

    

