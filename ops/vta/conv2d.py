import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm import topi

def conv2d_packed(ctx, data, kernel, strides, padding, dilation, out_dtype):
    assert dilation == (1, 1)
    if padding[0]:
        pad_data = topi.nn.pad(data, [0, 0, padding[0], padding[1], 0, 0], name="pad_data")
    else:
        pad_data = data
    assert len(data.shape) == 6
    assert len(kernel.shape) == 6
    oheight = topi.util.get_const_int((pad_data.shape[2] - kernel.shape[2]) // strides[0] + 1)
    owidth = topi.util.get_const_int((pad_data.shape[3] - kernel.shape[3]) // strides[1] + 1)
    oshape = (data.shape[0], kernel.shape[0], oheight, owidth, data.shape[4], kernel.shape[4])

    ishape = topi.util.get_const_tuple(data.shape)
    kshape = topi.util.get_const_tuple(kernel.shape)
    d_i = te.reduce_axis((0, kshape[2]), name="d_i")
    d_j = te.reduce_axis((0, kshape[3]), name="d_j")
    k_o = te.reduce_axis((0, ishape[1]), name="k_o")
    k_i = te.reduce_axis((0, ishape[-1]), name="k_i")
    hstride, wstride = strides
    res = te.compute(
        oshape,
        lambda b_o, c_o, i, j, b_i, c_i: te.sum(
            pad_data[b_o, k_o, i * hstride + d_i, j * wstride + d_j, b_i, k_i].astype(out_dtype)
            * kernel[c_o, k_o, d_i, d_j, c_i, k_i].astype(out_dtype),
            axis=[k_o, d_i, d_j, k_i],
        ),
        name="res",
        tag="conv2d_dense",
    )
    return res

