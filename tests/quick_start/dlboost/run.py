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
    if config.tuned == "":
        env.tune(task_name)

    if config.tuned == "":
        config.tuned = os.path.join(config.log_dir, 'records.txt')
    sch, args = task.apply_best(config.tuned)

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
    latency = evaluator(*tensors).mean * 1e3
    if latency > 0:
        print("PASS")
    return latency


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run cases in Heron.')
    parser.add_argument("-p", "--platform", choices=['tensorcore', 'dlboost'], type=str, default="dlboost", help="Hardware platform: currently tensorcore and dlboost.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file name: config files locate under configs folder and describe the param of test_cases")
    parser.add_argument("-t", "--tuned", type=str, default="", help="Test config file name: tuned parameters' configueration for direct use.")
    parser.add_argument("-d", "--device_id", type=int, default=0, help="Device index for measurement")
    parser.add_argument("-sid", "--start_id", type=int, default=0, help="Start case number in test cases")
    parser.add_argument("-eid", "--end_id", type=int, default=1e4, help="End case number in test cases")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show some information if enabled.")
    parser.add_argument("-m", "--max-trials", type =int, default=1e6, help="Max trials by command line")
    args = parser.parse_args()
    print("Heron device ", args.device_id)

    config = makeConfig(args)
    config.max_trials = min(args.max_trials, config.max_trials)
    config.device_id = args.device_id
    config.tuned = args.tuned
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
    cases = []
    for i in range(args.start_id, min(args.end_id + 1, len(config.cases))):
        if i >= 1:
            config.tuned = ''
        case_keys = list(config.cases.keys())
        op_name, case = config.cases[case_keys[i]]
        print(case)
        env = Env(measure_option, config)
        latency = run(op_name, case_keys[i], case + [config.in_dtype, config.out_dtype], env)
        cases.append([case, latency])
    for tup in cases:
        print("Case %s, latency %f ms."%(str(tup[0]), tup[1]))


