# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
# pylint: disable=no-value-for-parameter,import-outside-toplevel
"""Conv2D int8 schedule on x86"""

import tvm
from tvm import te
from tvm import autotvm
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.nn.conv2d import unpack_NCHWc_to_nchw
from tvm.topi.utils import simplify, get_const_tuple, traverse_inline
from tvm.topi import nn
from tvm.topi.nn.pad import pad
from tvm.topi.x86.tensor_intrin import *
from Heron.utils import get_divisable
    
def expandAndPad(inputs, stride, padding):
    padding_zero = tvm.tir.expr.const(0, inputs.dtype)
    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    assert isinstance(stride, tuple), "type(stride)={}".format(type(stride))
    assert len(stride) == 2

    zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, height, width, in_channel = inputs.shape
    H_O = (height - 1) * stride[0] + 1
    W_O = (width - 1) * stride[1] + 1
    return tvm.te.compute(
        (batch_size, H_O + padding[0] + padding[1], W_O + padding[2] + padding[3], in_channel),
        lambda b, h, w, c: tvm.te.if_then_else(
            tvm.te.all(
                h >= padding[0], h < H_O + padding[0], (h-padding[0]) % stride[0] == 0,
                w >= padding[2], w < width + padding[2], (w-padding[2]) % stride[1] == 0
            ),
            inputs[b, (h - padding[0]) // stride[0], (w - padding[2]) // stride[1], c],
            zero,
        ),
        name="Padding",
        tag = "injective,pad"
        )


def heron_conv2d_transposed_nhwc(ctx,
                      N, H, W, CI,
                      CO, KH, KW,
                      stride, padding, dilation,outpadding, w_format, in_dtype, out_dtype
                      ):
    assert w_format in ["OIhw4i16o4i"]
    split_co = 16
    split_ci = 4
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    a_shape = (N, H, W, CI)
    w_shape = (CO // split_co,  CI // (split_ci * split_ci), KH, KW, split_ci, split_co, split_ci)
    Input = te.placeholder(a_shape, name="input", dtype = 'uint8')
    Filter = te.placeholder(w_shape, name="filter", dtype = 'int8')
    # Reduction axes
    ic_1 = te.reduce_axis((0, CI // (split_ci * split_ci)), name="ic_1")
    ic_2 = te.reduce_axis((0, split_ci), name="ic_2")
    ic_3 = te.reduce_axis((0, split_ci), name="ic_3")
    ry = te.reduce_axis((0, KH), name="ry")
    rx = te.reduce_axis((0, KW), name="rx")
    dilated_KH = (KH - 1) * dilation + 1
    dilated_KW = (KW - 1) * dilation + 1
    H_O = (H - 1) * stride - 2 * padding + dilated_KH + outpadding
    W_O = (W - 1) * stride - 2 * padding + dilated_KW + outpadding
    PaddedInput = expandAndPad(Input, stride,
                               padding=(
                                   dilated_KH - 1 - padding,
                                   dilated_KH - 1 - padding + outpadding,
                                   dilated_KW - 1 - padding,
                                   dilated_KW - 1 - padding + outpadding,
                               )
                                )

    out = te.compute(
        (N, H_O, W_O, CO//split_co, split_co),
        lambda nn, yy, xx, oc_chunk, oc_block: te.sum(
            PaddedInput[
                nn, 
                yy * stride_h + ry * dilation_h,
                xx * stride_w + rx * dilation_w,
                ic_1 * split_ci * split_ci + ic_2 * split_ci + ic_3
            ].astype("int32")
            * Filter[
                oc_chunk,
                ic_1,
                ry,
                rx,
                ic_2,
                oc_block,
                ic_3].astype("int32"),
            axis=[ry, rx, ic_1, ic_2, ic_3],
        ),
        name="conv",
    )

    # pass information of tensorcore strategy to optimizer
    info = {
            'name' : 'x86_dot',
            'stage_orgination' : [('conv', 'addCacheWriteGlobal')],
            'com' : ('conv.global', (4, 4)),
            'intrin': [dot_16x1x16_uint8_int8_int32_cascadelake, 16, 4]
            }
    outs = [out]
    ctx.set_info(info)
    axis_map = {
            'conv.global' : ['s_0', 's_1', 's_2', 's_3', 's_4'],
            }
    ctx.set_axis_map(axis_map)
    return outs


