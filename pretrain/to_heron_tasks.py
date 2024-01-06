from utils import *
import os
import time
import math
import argparse
from ortools.sat.python import cp_model

import tvm
import tvm.autotvm as autotvm 

import numpy as np

from Heron.environment import Env
import Heron.runner as HeronRunner
from Heron.config import configFromFile
import Heron.ops.cuda as heron_cuda
import Heron.ops.x86 as heron_cpu
from Heron.config import Config


def makeConfig(args):
    config = Config()
    config.runner_timeout = 50 
    config.build_timeout = 50 
    config.runner_number = 3
    if args.platform == "tensorcore":
        config.runner_repeat = 2
        config.target_name = 'cuda'
        config.codegen_type = 'GPU_TENSOR_CORE'
        config.get_op = heron_cuda.getOpFromName 
        config.in_dtype = "float16"
        config.out_dtype = "float16"
    elif args.platform == "dlboost":
        config.target_name = "llvm --device=cpu -mcpu=cascadelake"
        config.codegen_type = "CPU" 
        config.get_op = heron_cpu.getOpFromName 
        config.in_dtype = "int8"
        config.out_dtype = "int32"
    return config

def run(op_name, task_name, params, env):
    config = env.config
    target = tvm.target.Target(config.target_name)
    op = config.get_op(op_name)
    task = env.createTask(task_name, op, params, target)
    task.device_id = config.device_id
    env.tune(task_name)


def taskConvert(task):
    name, tensors = task
    op_name = None
    args = None
    if name in ['conv2d_nchw.cuda', 'conv2d_nchw_winograd']:
       op_name = "c2d"
       inp, wei, stride, pad, dilation, _ = tensors
       _, inp_shape, _ = inp 
       _, wei_shape, _ = wei 
       N, CI, H, W = inp_shape
       CO, CI, KH, KW = wei_shape
       args = [N, H, W, CI, CO, KH, KW, stride, pad, dilation[0]]
    elif name in ['dense_small_batch.gpu', 'dense_large_batch.gpu']:
       op_name = "gemm"
       A, B, _, _ = tensors 
       _, A_shape, _ = A
       _, B_shape, _ = B
       M, K = A_shape
       N, _ = B_shape
       args = [M, K, N]
    elif name in ['conv3d_ncdhw.cuda', 'conv3d_ncdhw_winograd.cuda']:
       op_name = "c3d"
       inp, wei, stride, pad, dilation, groups, _ = tensors
       _, inp_shape, _ = inp 
       _, wei_shape, _ = wei 
       N, CI, D, H, W = inp_shape
       CO, CI, KD, KH, KW = wei_shape
       args = [N, D, H, W, CI, CO, KD, KH, KW, stride, pad, dilation[0]]
    elif name == "batch_matmul.cuda":
       op_name = "bmm"
       A, B, _, _, ta, tb = tensors
       _, A_shape, _ = A
       _, B_shape, _ = B
       if not ta:
           batch, M, K = A_shape
       else:
           batch, K, M = A_shape
       if not tb:
           batch, _, N = A_shape
       else:
           batch, N, _ = A_shape

       args = [batch, M, K, N]
    return op_name, args

def genHeronTasks():
    files, fpaths = list_files_in_folder("workloads") 
    res = []
    existed = set()
    for fpath in fpaths:
        data = read_json_file(fpath)
        net_name = data["netname"]
        args = data["args"]
        tasks = data["tasks"]
        for t in tasks:
            tname, targs = taskConvert(t)
            if not tname:
                continue
            key = tname + str(targs)
            if key not in existed:
                res.append([tname, targs])
                existed.add(key)
    for r in res:
        print(r)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dataset for training.')
    parser.add_argument("-p", "--platform", choices=['tensorcore', 'dlboost'], type=str, default="tensorcore", help="Hardware platform: currently tensorcore and dlboost.")
    parser.add_argument("-sid", "--start_id", type=int, default=0, help="Start case number in test cases")
    parser.add_argument("-eid", "--end_id", type=int, default=1e4, help="End case number in test cases")
    parser.add_argument("-dev", "--dev_id", type=int, default=0, help="device id")
    args = parser.parse_args()

    config = makeConfig(args)
    config.device_id = args.dev_id
    config.max_trials = 2000
    config.opt_method = "CRANDS"
    if args.platform == "dlboost":
        flush=True
    else:
        flush = False
    runner = HeronRunner.LocalRunner(
                                     number=config.runner_number,
                                     repeat=config.runner_repeat,
                                     min_repeat_ms=500,
                                     timeout=config.runner_timeout, enable_cpu_cache_flush = flush)
    measure_option = autotvm.measure_option(
                builder = autotvm.LocalBuilder(timeout=config.build_timeout),
                runner = runner
            )  
    tasks = genHeronTasks()
    print(len(tasks))
    for i in range(args.start_id, min(args.end_id + 1, len(tasks))):
        op_name, case = tasks[i]
        task_name = "task_%d_"%i + op_name + str(case)
        config.out_name = "OPS/" + op_name
        if os.path.exists(os.path.join(config.out_name, task_name)):
            print("%s exists, continue"%task_name)
            continue
        print(task_name)
        env = Env(measure_option, config)
        try:
            run(op_name, task_name, case + [config.in_dtype, config.out_dtype], env)
        except Exception as ex:
            print(ex)
