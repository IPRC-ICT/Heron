import numpy as np
import tvm
from tvm import topi
import tvm.topi.testing
from tvm import te
from tvm.contrib.pickle_memoize import memoize
from tvm.contrib import nvcc
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple
from tvm.autotvm import task
from tvm.autotvm.task import ConfigEntity
from tvm.target import Target

import tvm.autotvm as autotvm
import tvm.testing
import os
import shutil
import json
try:  # convert unicode to str for python2
    _unicode = unicode
except NameError:
    _unicode = ()

try:
    _long = long
except NameError:
    _long = int


target_name = 'cuda'
dtype = "float16"
def clean_json_to_python(x):
    """1. Convert all list in x to tuple (hashable)
    2. Convert unicode to str for python2
    """
    if isinstance(x, list):
        return tuple([clean_json_to_python(a) for a in x])
    if isinstance(x, _unicode):
        return str(x)
    if isinstance(x, (_long, int)):
        return int(x)
    return x

def travel_for_shared_size(stmt):
    res = []
    def func(ir):
        if isinstance(ir, tvm.tir.stmt.BufferStore):
            res.append(ir)

    tvm.tir.stmt_functor.post_order_visit(stmt["main"].body, func)
    read_size= int(res[0].buffer.shape[0])
    write_size= int(res[2].value.buffer.shape[0])
    #read_size=0
    #write_size=0
    #for ir in res:
    #    scope = str(ir.buffer.scope())
    #    if scope == "shared":
    #        size = int(ir.buffer.shape[0])
    #        read_size += size
    #    else:
    #        size = int(ir.value.buffer.shape[0])
    #        write_size += size
    return read_size, write_size


def run(args):
    M, K, N = args
    task_name = 'dense'
    for x in args:
        task_name += '_' + str(x)
    log_file = "%s.log"%(task_name)
    data = []
    with open(log_file) as f:
        for row in f:
            if row and not row.startswith("#"):
                row = json.loads(row)
                latency = row['result'][0][0]
                if latency == 1e9:
                    continue
                tgt, task_name, task_args, task_kwargs = row["input"]
                tgt = str(tgt)
                tgt = Target(str(tgt))
                tsk = task.Task(clean_json_to_python(task_name), clean_json_to_python(task_args))
                config = ConfigEntity.from_json_dict(row["config"])
                s, arg_bufs = tsk.instantiate(config)
                stmt = tvm.lower(s, arg_bufs)
                size1, size2 = travel_for_shared_size(stmt)
                data.append([size1, size2, latency])
    npdata = np.array(data)
    np.savetxt("tvm.txt", npdata)



if __name__ == "__main__":
    dev = tvm.device(target_name, 0)
    if not tvm.testing.device_enabled(target_name):
        print("Skip because %s is not enabled" % target_name)
    if not nvcc.have_tensorcore(dev.compute_version):
        print("skip because gpu does not support Tensor Cores")
    print("Running on target: %s" % target_name)
    arg = (1024, 1024, 1024)
    run(arg)

