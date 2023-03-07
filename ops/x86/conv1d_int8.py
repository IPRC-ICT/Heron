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
from tvm.topi.nn.utils import get_pad_tuple1d
from tvm.topi.nn.pad import pad
from tvm.topi.x86.tensor_intrin import *
from Heron.utils import get_divisable
def heron_conv1d_nwc(ctx,
                     batch,
                     in_width,
                     co,
                     ci,
                     kernel,
                     stride,
                     padding,
                     dilation,
                     w_format,
                     in_dtype, out_dtype
                      ):
    assert w_format in ["OwI64o4i", "OwI48o4i", "OwI32o4i"]
    split_co = int(w_format.split('I')[1].split('o')[0])
    split_ci = int(w_format.split('o')[1].split('i')[0])

    a_shape = (batch, in_width, ci)
    w_shape = (co // split_co,  kernel, ci // split_ci, split_co, split_ci)
    Input = te.placeholder(a_shape, name="input", dtype = 'uint8')
    Filter = te.placeholder(w_shape, name="filter", dtype = 'int8')
    # Reduction axes
    ic_chunk = te.reduce_axis((0, ci // split_ci), name="ic_chunk")
    ic_block = te.reduce_axis((0, split_ci), name="ic_block")
    rx = te.reduce_axis((0, kernel), name="rx")

    dilated_kernel_size = (kernel - 1) * dilation + 1
    pad_left, pad_right = get_pad_tuple1d(padding, (dilated_kernel_size,))
    out_width = simplify((in_width - dilated_kernel_size + pad_left + pad_right) // stride + 1)

    # pad
    if padding != 0:
        pad_before = [0, pad_left, 0]
        pad_after = [0, pad_right, 0]
        PaddedInput = pad(Input, pad_before, pad_after, name="pad")
    else:
        PaddedInput = Input

    out = te.compute(
        (batch, out_width, co//split_co, split_co),
        lambda nn, xx, oc_chunk, oc_block: te.sum(
            PaddedInput[
                nn, 
                xx * stride + rx * dilation,
                ic_chunk * split_ci + ic_block
            ].astype("int32")
            * Filter[
                oc_chunk,
                rx,
                ic_chunk,
                oc_block,
                ic_block].astype("int32"),
            axis=[rx, ic_chunk, ic_block],
        ),
        name="conv",
    )

    # pass information of tensorcore strategy to optimizer
    info = {
            'name' : 'x86_dot',
            'stage_orgination' : [('conv', 'addCacheWriteGlobal')],
            'com' : ('conv.global', (3, 2)),
            'intrin': [dot_16x1x16_uint8_int8_int32_cascadelake, 16, 4]
            }
    outs = [out]
    ctx.set_info(info)
    axis_map = {
            'pad' : ['s_0', 's_1', 'r_1'],
            'conv.global' : ['s_0', 's_1', 's_2', 's_3'],
            }
    ctx.set_axis_map(axis_map)
    return outs


