import torch
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def makeConfig(args):
    config = configFromFile(args.config) 
    if args.platform == "tensorcore":
        config.target_name = 'cuda'
        config.codegen_type = 'GPU_TENSOR_CORE'
        config.get_op = heron_cuda.getOpFromName 
    elif args.platform == "dlboost":
        config.target_name = "llvm --device=cpu -mcpu=cascadelake"
        config.codegen_type = "CPU" 
        config.get_op = heron_cpu.getOpFromName 
    return config

def run(op_name, task_name, params, env):
    config = env.config
    target = tvm.target.Target(config.target_name)
    op = config.get_op(op_name)
    task = env.createTask(task_name, op, params, target)
    task.device_id = config.device_id
    env.tune(task_name)

    sch, args = task.apply_best(os.path.join(config.log_dir, 'records.txt'))
    print("Reported time: %f ms"%(math.exp(-task.knob_manager.runtime_perf)* 1e3))
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))

    # verify
    # scheduled
    with tvm.target.Target(config.target_name):
        s, args = task.instantiate(task.knob_manager)
        func = tvm.build(s, args)
    if config.target_name == "cuda": 
        dev = tvm.cuda(config.device_id)
    elif "cpu" in config.target_name:
        dev = tvm.cpu()

    tensors = []
    for T in args:
        shape = [int(x) for x in T.shape]
        dtype = T.dtype
        Tnp = np.random.uniform(size=shape).astype(dtype)
        Ttvm = tvm.nd.array(Tnp, dev)
        tensors.append(Ttvm)

    evaluator = func.time_evaluator(func.entry_name, dev,
            number=config.runner_number, repeat=config.runner_repeat, min_repeat_ms = 500)
    return (evaluator(*tensors).mean * 1e3)


def runArgs(args):
    config = makeConfig(args)
    config.device_id = args.device_id
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
    res = []
    for i in range(args.start_id, min(args.end_id + 1, len(config.cases))):
        case_keys = list(config.cases.keys())
        op_name, case = config.cases[case_keys[i]]
        print(case)
        env = Env(measure_option, config)
        latency = run(op_name, case_keys[i], case + [config.in_dtype, config.out_dtype], env)
        res.append(latency)
    return res 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run cases in Heron.')
    parser.add_argument("-p", "--platform", choices=['tensorcore', 'dlboost'], type=str, default="tensorcore", help="Hardware platform: currently tensorcore and dlboost.")
    parser.add_argument("-r", "--repeat", type=int, default=5, help="Repeat num of experiments")
    parser.add_argument("-d", "--device_id", type=int, default=0, help="Device index for measurement")
    parser.add_argument("-sid", "--start_id", type=int, default=0, help="Start case number in test cases")
    parser.add_argument("-eid", "--end_id", type=int, default=1e4, help="End case number in test cases")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show some information if enabled.")
    args = parser.parse_args()
    print("Heron device ", args.device_id)
    method_paths = [
            "configs/cga.json",
            "configs/cga1.json",
            "configs/ga1.json",
            "configs/ga2.json",
            "configs/ga3.json",
            ]
    plt.figure()
    x = np.array([1024, 2048, 3072, 4096, 5120])
    cga_res = None
    for path in method_paths:
        method = path.split('/')[1].split('.')[0]
        args.config = path
        res = 0
        for i in range(args.repeat):
            res += np.array(runArgs(args))
        res = res / args.repeat
        if method == "cga":
            cga_res = res
        plt.plot(x, cga_res / res, label=method)
    plt.legend()
    plt.savefig('figure13.png')
    plt.close()


        



