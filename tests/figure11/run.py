import os
import json
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
from tvm.autotvm.measure.measure import create_measure_batch
from tvm import auto_scheduler

def travel_for_shared_size(stmt):
    res = []
    def func(ir):
        if isinstance(ir, tvm.tir.stmt.BufferStore):
            res.append(ir)

    tvm.tir.stmt_functor.post_order_visit(stmt["main"].body, func)
    read_size= int(res[0].buffer.shape[0])
    write_size= int(res[2].value.buffer.shape[0])
   # for ir in res:
   #     scope = str(ir.buffer.scope())
   #     if scope == "shared":
   #         size = int(ir.buffer.shape[0])
   #         read_size += size
   #     else:
   #         size = int(ir.value.buffer.shape[0])
   #         write_size += size
    return read_size, write_size


def init_env(env):
    env.init_dir()
    env.runner.measure_batch = create_measure_batch(env.task, env.runner.measure_option)

def getMems(env, param):
    env.task.knob_manager.solved_knob_vals = param
    env.task.knob_manager.is_building=False
    s, args = env.task.instantiate(env.task.knob_manager)
    stmt = tvm.lower(s, args)
    return travel_for_shared_size(stmt)

def makeConfig(args):
    config = configFromFile(args.config) 
    if args.platform == "tensorcore":
        config.target_name = 'cuda'
        config.codegen_type = 'GPU_TENSOR_CORE'
        config.get_op = heron_cuda.getOpFromName 
    elif args.platform == "dlboost":
        # Remove this if you are sure about your -mcpu argument
        assert 0 
        config.target_name = "llvm --device=cpu -mcpu=cascadelake"
        config.codegen_type = "CPU" 
    return config

def run(op_name, task_name, params, env, path):
    config = env.config
    target = tvm.target.Target(config.target_name)
    op = config.get_op(op_name)
    task = env.createTask(task_name, op, params, target)
    task.device_id = 0
    data = []
    for row in open(path):
        json_dict = json.loads(row)
        perf = json_dict['perf']
        if perf == 0:
            continue
        param = json_dict['param']
        mem_a, mem_b = getMems(env, param)
        data.append([mem_a, mem_b, perf])
    npdata = np.array(data)
    np.savetxt("heron.txt", npdata)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run cases in Heron.')
    parser.add_argument("-p", "--platform", choices=['tensorcore', 'dlboost'], type=str, required=True, default="tensorcore", help="Hardware platform: currently tensorcore and dlboost.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file name: config files locate under configs folder and describe the param of test_cases")
    parser.add_argument("-t", "--tuned", type=str, required=True, help="Test config file name: tuned parameters' configueration for direct use.")
    args = parser.parse_args()

    config = makeConfig(args)
    config.device_id = 0
    config.tuned = args.tuned
    runner = HeronRunner.LocalRunner(
                                     number=config.runner_number,
                                     repeat=config.runner_repeat,
                                     min_repeat_ms=500,
                                     timeout=config.runner_timeout)
    measure_option = autotvm.measure_option(
                builder = autotvm.LocalBuilder(timeout=config.build_timeout),
                runner = runner
            )  
    case_keys = list(config.cases.keys())
    op_name, case = config.cases[case_keys[0]]
    print(case)
    env = Env(measure_option, config)
    run(op_name, case_keys[0], case + [config.in_dtype, config.out_dtype], env, args.tuned)


