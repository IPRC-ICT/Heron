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
from tvm.topi.nn.utils import get_pad_tuple3d
from tvm.topi.utils import simplify, get_const_tuple, traverse_inline
from tvm.topi import nn
from tvm.topi.nn.pad import pad
from tvm.topi.x86.tensor_intrin import *
from Heron.utils import get_divisable
def heron_conv3d_ndhwc(ctx,
                      batch,
                      in_depth,
                      in_height,
                      in_width,
                      num_filter,
                      in_channel,
                      kernel_d,
                      kernel_h,
                      kernel_w,
                      stride,
                      padding,
                      dilation,
                      w_format,
                      in_dtype, out_dtype
                      ):
    assert w_format in ["OdhwI64o4i", "OdhwI48o4i", "OdhwI32o4i"]
    split_co = int(w_format.split('I')[1].split('o')[0])
    split_ci = int(w_format.split('o')[1].split('i')[0])
    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_d = dilation_h = dilation_w = dilation
    else:
        dilation_d, dilation_h, dilation_w = dilation

    a_shape = (batch, in_depth, in_height, in_width, in_channel)
    w_shape = (num_filter // split_co,  kernel_d, kernel_h, kernel_w, in_channel // split_ci, split_co, split_ci)
    Input = te.placeholder(a_shape, name="input", dtype = 'uint8')
    Filter = te.placeholder(w_shape, name="filter", dtype = 'int8')
    # Reduction axes
    ic_chunk = te.reduce_axis((0, in_channel // split_ci), name="ic_chunk")
    ic_block = te.reduce_axis((0, split_ci), name="ic_block")
    rz = te.reduce_axis((0, kernel_d), name="rz")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    # pad
    dilated_kernel_d = (kernel_d - 1) * dilation_d + 1
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_front, pad_top, pad_left, pad_back, pad_down, pad_right = get_pad_tuple3d(
        padding, (dilated_kernel_d, dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = num_filter
    out_depth = simplify((in_depth - dilated_kernel_d + pad_front + pad_back) // stride_d + 1)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_front, pad_top, pad_left, 0]
    pad_after = [0, pad_back, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    if padding != 0:
        PaddedInput = pad(Input, pad_before, pad_after, name="pad")
    else:
        PaddedInput = Input

    out = te.compute(
        (batch, out_depth, out_height, out_width, out_channel//split_co, split_co),
        lambda nn, zz, yy, xx, oc_chunk, oc_block: te.sum(
            PaddedInput[
                nn, 
                zz * stride_d + ry * dilation_d,
                yy * stride_h + ry * dilation_h,
                xx * stride_w + rx * dilation_w,
                ic_chunk * split_ci + ic_block
            ].astype("int32")
            * Filter[
                oc_chunk,
                rz,
                ry,
                rx,
                ic_chunk,
                oc_block,
                ic_block].astype("int32"),
            axis=[rz, ry, rx, ic_chunk, ic_block],
        ),
        name="conv",
    )

    # pass information of tensorcore strategy to optimizer
    info = {
            'name' : 'x86_dot',
            'stage_orgination' : [('conv', 'addCacheWriteGlobal')],
            'com' : ('conv.global', (5, 4)),
            'intrin': [dot_16x1x16_uint8_int8_int32_cascadelake, 16, 4]
            }
    outs = [out]
    ctx.set_info(info)
    axis_map = {
            'pad' : ['s_0', 's_1', 's_2', 's_3', 'r_3'],
            'conv.global' : ['s_0', 's_1', 's_2', 's_3', 's_4', 's_5'],
            }
    ctx.set_axis_map(axis_map)
    return outs


