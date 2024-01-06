import threading
import logging
import warnings
import tvm
from tvm import relay
from tvm import topi
from tvm.target import Target
from tvm.autotvm.task.topi_integration import TaskExtractEnv
from tvm.autotvm.task.relay_integration import _lower
from tvm.autotvm.task.dispatcher import DispatchContext, FallbackContext
logger = logging.getLogger("autotvm")

def extract_from_programs(mods, params, target, target_host = None, ops=None):
    env = TaskExtractEnv.get()

    # merge target and target host
    target, target_host = Target.check_and_update_host_consist(target, target_host)

    # run compiler to collect all TOPI calls during compilation
    env.reset(ops)
    with env:
        # disable logger temporarily
        old_state = logger.disabled
        logger.disabled = True

        for mod, param in zip(mods, params):
            if isinstance(mod, relay.function.Function):
                mod = tvm.IRModule.from_expr(mod)
            assert isinstance(
                mod, tvm.IRModule
            ), "only support relay Module or Function to be tuned"
            relay.backend.te_compiler.get().clear()
            # wrap build call in thread to avoid multiprocessing problems
            build_thread = threading.Thread(target=_lower, args=(mod, target, param))
            build_thread.start()
            build_thread.join()
            relay.backend.te_compiler.get().clear()
            # Clear the warning message cache in FallbackContext
            if isinstance(DispatchContext.current, FallbackContext):
                DispatchContext.current.memory = {}
                DispatchContext.warning_messages = set()

        logger.disabled = old_state

    return env.get_tasks()
