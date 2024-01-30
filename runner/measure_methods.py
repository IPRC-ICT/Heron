import tvm.autotvm as autotvm
from tvm.autotvm.measure.measure_methods import *
import numpy as np

class LocalRunner(autotvm.LocalRunner):
    def run(self, measure_inputs, build_results):
        results = []
        remote_kwargs = dict(
            device_key=self.key,
            host=self.host,
            port=self.port,
            priority=self.priority,
            timeout=self.timeout,
        )

        for i in range(0, len(measure_inputs), self.n_parallel):
            futures = []
            for measure_inp, build_res in zip(
                measure_inputs[i : i + self.n_parallel], build_results[i : i + self.n_parallel]
            ):
                module_loader = (
                    self.module_loader
                    if self.module_loader is not None
                    else default_module_loader()
                )
                ret = self.executor.submit(
                    my_run_through_rpc,
                    measure_inp,
                    build_res,
                    self.number,
                    self.repeat,
                    self.min_repeat_ms,
                    self.cooldown_interval,
                    remote_kwargs,
                    self.enable_cpu_cache_flush,
                    module_loader,
                )
                futures.append(ret)

            for future in futures:
                try:
                    res = future.result()
                except Exception as ex:
                    res = None
                if res == None or isinstance(res, Exception):  # executor error or timeout
                    results.append(
                        MeasureResult(
                            (str(res),), MeasureErrorNo.RUN_TIMEOUT, self.timeout, time.time()
                        )
                    )
                else:
                    results.append(res)

        return results
class RPCRunner(autotvm.RPCRunner):
    def run(self, measure_inputs, build_results):
        results = []
        remote_kwargs = dict(
            device_key=self.key,
            host=self.host,
            port=self.port,
            priority=self.priority,
            timeout=self.timeout,
        )

        for i in range(0, len(measure_inputs), self.n_parallel):
            futures = []
            for measure_inp, build_res in zip(
                measure_inputs[i : i + self.n_parallel], build_results[i : i + self.n_parallel]
            ):
                module_loader = (
                    self.module_loader
                    if self.module_loader is not None
                    else default_module_loader()
                )
                ret = self.executor.submit(
                    my_run_through_rpc,
                    measure_inp,
                    build_res,
                    self.number,
                    self.repeat,
                    self.min_repeat_ms,
                    self.cooldown_interval,
                    remote_kwargs,
                    self.enable_cpu_cache_flush,
                    module_loader,
                )
                futures.append(ret)

            for future in futures:
                res = future.result()
                if isinstance(res.costs, tuple):
                    print(res.costs[0])
                print("=============================================")
                if isinstance(res, Exception):  # executor error or timeout
                    results.append(
                        MeasureResult(
                            (str(res),), MeasureErrorNo.RUN_TIMEOUT, self.timeout, time.time()
                        )
                    )
                else:
                    results.append(res)

        return results

def my_run_through_rpc(
    measure_input,
    build_result,
    number,
    repeat,
    min_repeat_ms,
    cooldown_interval,
    remote_kwargs,
    enable_cpu_cache_flush=False,
    module_loader=None,
):
    if isinstance(build_result, MeasureResult):
        return build_result

    tic = time.time()
    errno = MeasureErrorNo.NO_ERROR
    task = measure_input.task
    if hasattr(measure_input.task, "device_id"):
        device_id = measure_input.task.device_id
    else:
        device_id = 0
    try:
        # upload built module
        with module_loader(remote_kwargs, build_result) as (remote, mod):
            dev = remote.device(str(measure_input.target), device_id)

            # Limitation:
            # We can not get PackFunction directly in the remote mode as it is wrapped
            # under the std::function. We could lift the restriction later once we fold
            # the PackedFunc as an object. Currently, we pass function name to work
            # around it.
            f_prepare = "cache_flush_cpu_non_first_arg" if enable_cpu_cache_flush else ""
            time_f = mod.time_evaluator(
                mod.entry_name,
                dev,
                number=number,
                repeat=repeat,
                min_repeat_ms=min_repeat_ms,
                f_preproc=f_prepare,
            )

            if task.not_verify_correctness:
                try:
                    random_fill = remote.get_function("tvm.contrib.random.random_fill")
                except AttributeError:
                    raise AttributeError(
                        "Please make sure USE_RANDOM is ON in the config.cmake " "on the remote devices"
                    )
                args = [nd.empty(x[0], x[1], dev) for x in build_result.arg_info]
                if "scatter" not in measure_input.task.name:
                    # the index tensor of scatter op cannot be randomly initialized
                    for arg in args:
                        random_fill(arg)
            else:
                args = [nd.array(x, dev) for x in task.ref_input]

            dev.sync()
            
            costs = time_f(*args).results

        # Use median
        if len(costs) > 2:  
            costs = list(costs)
            costs.sort()
            med_idx = int(len(costs) // 2)
            costs = costs[med_idx]
        # check correctness
        if not task.not_verify_correctness:
            expected = task.ref_output
            output = args[-1].asnumpy()
            if not np.allclose(expected, output, rtol = task.rtol):
                errno = 100
    except TVMError as exc:
        msg = str(exc)
        if "Stack trace returned" in msg:
            msg = msg[: msg.index("Stack trace returned")]
        if "CUDA Source" in msg:
            msg = msg[: msg.index("CUDA Source")]
        costs = (RuntimeError(msg[:2048]),)
        errno = MeasureErrorNo.RUNTIME_DEVICE
    tstamp = time.time()
    time.sleep(cooldown_interval)
    return MeasureResult(costs, errno, tstamp - tic + build_result.time_cost, tstamp)
