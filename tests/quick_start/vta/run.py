import os
import time
import math
import argparse
from ortools.sat.python import cp_model

import tvm
import tvm.autotvm as autotvm 
from tvm import rpc

import numpy as np

from Heron.environment import Env
import Heron.runner as HeronRunner
from Heron.config import configFromFile
import Heron.ops.cuda as heron_cuda
import Heron.ops.x86 as heron_cpu
import Heron.ops.vta as heron_vta
from tvm.contrib import utils
import vta
from vta.testing import simulator
vta_env = vta.get_env()


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
    elif args.platform == 'vta':
        target = vta_env.target
        config.target_name = str(target)
        config.codegen_type = "VTA"
        config.get_op = heron_vta.getOpFromName
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

    # evaluate with tuning history
    if vta_env.TARGET != "sim":
        # Get remote from fleet node
        remote = autotvm.measure.request_remote(
            vta_env.TARGET, tracker_host, tracker_port, timeout=10000
        )
        # Reconfigure the JIT runtime and FPGA.
        vta.reconfig_runtime(remote)
        vta.program_fpga(remote, bitstream=None)
    else:
        # In simulation mode, host the RPC server locally.
        remote = rpc.LocalSession()

    # Build
    mod = vta.build( sch, args,
        target=tvm.target.Target(target, host=vta_env.target_host),
        name="op",)
    temp = utils.tempdir()
    mod.export_library(temp.relpath("op.tar"))
    remote.upload(temp.relpath("op.tar"))
    f = remote.load_module("op.tar")
    dev = remote.device(str(target))

    # Get the remote device context
    tensors = []
    for T in args:
        shape = [int(x) for x in T.shape]
        dtype = T.dtype
        Tnp = np.random.uniform(size=shape).astype(dtype)
        Ttvm = tvm.nd.array(Tnp, dev)
        tensors.append(Ttvm)
    time_f = f.time_evaluator("op", dev, number=4)

    # In vta sim mode, collect simulator runtime statistics
    cost = None
    if vta_env.TARGET in ["sim", "tsim"]:
        # Check if we're in local RPC mode (allows us to rebuild the
        # runtime on the fly when varying the VTA designs)
        local_rpc = int(os.environ.get("VTA_LOCAL_SIM_RPC", "0"))
        if local_rpc:
            if env.TARGET == "sim":
                remote.get_function("vta.simulator.profiler_clear")()
            else:
                remote.get_function("vta.tsim.profiler_clear")()
            cost = time_f(*tensors)
        else:
            simulator.clear_stats()
            cost = time_f(*tensors)
    else:
        cost = time_f(*tensors)
    print(cost)
    
    return cost.mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run cases in Heron.')
    parser.add_argument("-p", "--platform", choices=['tensorcore', 'dlboost', 'vta'], type=str, default="vta", help="Hardware platform: tensorcore , dlboost, and vta.")
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
    tracker_host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1")
    tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
    runner = HeronRunner.RPCRunner(
                                     vta_env.TARGET,
                                     host='127.0.0.1',
                                     port=9190,
                                     number=config.runner_number,
                                     timeout=config.runner_timeout,
                                     module_loader = None if vta_env.TARGET in ["sim"] else vta.module_loader(),
                                     )
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


