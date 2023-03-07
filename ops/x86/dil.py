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
from tvm.topi.utils import simplify, get_const_tuple, traverse_inline
from tvm.topi import nn
from tvm.topi.nn.pad import pad
from tvm.topi.x86.tensor_intrin import *
from Heron.utils import get_divisable
def heron_dil_nhwc(ctx,
                      N, H, W, CI,
                      CO, KH, KW,
                      stride, padding, dilation, w_format, in_dtype, out_dtype
                      ):
    assert w_format in ["OhwI64o4i", "OhwI48o4i", "OhwI32o4i"]
    split_co = int(w_format.split('I')[1].split('o')[0])
    split_ci = int(w_format.split('o')[1].split('i')[0])
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    a_shape = (N, H, W, CI)
    w_shape = (CO // split_co,  KH, KW, CI // split_ci, split_co, split_ci)
    Input = te.placeholder(a_shape, name="input", dtype = 'uint8')
    Filter = te.placeholder(w_shape, name="filter", dtype = 'int8')
    # Reduction axes
    ic_chunk = te.reduce_axis((0, CI // split_ci), name="ic_chunk")
    ic_block = te.reduce_axis((0, split_ci), name="ic_block")
    ry = te.reduce_axis((0, KH), name="ry")
    rx = te.reduce_axis((0, KW), name="rx")
    # pad
    dilated_KH = (KH - 1) * dilation_h + 1
    dilated_KW = (KW - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_KH, dilated_KW)
    )
    out_channel = CO
    out_height = simplify((H - dilated_KH + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((W - dilated_KW + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    if padding != (0,0):
        PaddedInput = pad(Input, pad_before, pad_after, name="pad")
    else:
        PaddedInput = Input

    out = te.compute(
        (N, out_height, out_width, out_channel//split_co, split_co),
        lambda nn, yy, xx, oc_chunk, oc_block: te.sum(
            PaddedInput[
                nn, 
                yy * stride_h + ry * dilation_h,
                xx * stride_w + rx * dilation_w,
                ic_chunk * split_ci + ic_block
            ].astype("int32")
            * Filter[
                oc_chunk,
                ry,
                rx,
                ic_chunk,
                oc_block,
                ic_block].astype("int32"),
            axis=[ry, rx, ic_chunk, ic_block],
        ),
        name="conv",
    )

    # pass information of tensorcore strategy to optimizer
    info = {
            'name' : 'x86_dot',
            'stage_orgination' : [('conv', 'addCacheWriteGlobal')],
            'com' : ('conv.global', (4, 3)),
            'intrin': [dot_16x1x16_uint8_int8_int32_cascadelake, 16, 4]
            }
    outs = [out]
    ctx.set_info(info)
    axis_map = {
            'pad' : ['s_0', 's_1', 's_2', 'r_2'],
            'conv.global' : ['s_0', 's_1', 's_2', 's_3', 's_4'],
            }
    ctx.set_axis_map(axis_map)
    return outs


