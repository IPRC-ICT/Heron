import time
from Heron.task.task import Task
from Heron.perf.perfBuffer import perfBuffer
from Heron.utils import *
from Heron.runner.runner import Runner
from Heron.schedule.sched_op import get_op_methods
from .tuner import *
from tvm.autotvm.measure.measure import create_measure_batch, MeasureInput

class Env:
    def __init__(self, measure_option, config):
        self.config = config
        self.runner = Runner(measure_option) 
        op_methods = get_op_methods(config.codegen_type)
        self.num_ops = len(op_methods.keys())
        self.build_kwargs = None
        self.task = None

    def init_dir(self):
        if os.path.exists(self.config.log_dir):
            shutil.rmtree(self.config.log_dir)
        mkdir(self.config.log_dir)

    def get_build_kwargs(self, task):
        get = create_kwargs_ctx(task, self.runner.measure_option)
        build_kwargs = get()
        del get
        self.build_kwargs = build_kwargs
        print(self.build_kwargs)

    def tune(self, task_name, pretrained = False):
        self.init_dir()
        start = time.time()
        self.task.make_stage_schedules()
        self.runner.measure_batch = create_measure_batch(self.task, self.runner.measure_option)
        self.dump_schedule()
        if pretrained:
            res = self.tuner.run_with_pretrained(self)
        else:
            res = self.tuner.run(self)
        print("Heron time spent ", time.time() - start)
        del self.runner.measure_batch
        return res

    def dump_schedule(self):
        path = os.path.join(self.config.log_dir, 'schedule.py')
        with open(path, 'w') as f:
            strs = self.task.sched_desc
            f.write(strs)


    def createTask(self, name, opfunc, args, target,
                    target_host = None, dump_const_desc = False):
        assert self.task == None
        self.config.task_name = name
        task = Task(name, opfunc, args, target, target_host)
        task.knob_manager.dump_descs = dump_const_desc
        if self.build_kwargs == None:
            self.get_build_kwargs(task)
        task.build_kwargs = self.build_kwargs
        task.config = self.config
        self.task = task

        # initialize
        self.config.setEnv(self)
        if self.config.opt_method == 'CRAND':
            self.tuner = CRandTuner(self.config)
        elif self.config.opt_method == 'CRANDS':
            self.tuner = CRandSampler(self.config)
        elif self.config.opt_method == 'CGA':
            self.tuner = CGATuner(self.config)
        elif self.config.opt_method == 'RCGA':
            self.tuner = RCGATuner(self.config)
        elif self.config.opt_method == 'GA':
            self.tuner = GATuner(self.config)
        elif self.config.opt_method == 'SRGA':
            self.tuner = SRGATuner(self.config)
        elif self.config.opt_method == 'SDGA':
            self.tuner = SDGATuner(self.config)
        elif self.config.opt_method == 'IDEA':
            self.tuner = IDEATuner(self.config)
       #elif self.config.opt_method == 'CSA':
       #    self.tuner = ConstrainedRandomWalkerSATuner(self.config)
        elif self.config.opt_method == 'SA':
            self.tuner = RandomWalkSATuner(self.config)
        else:
            raise ValueError('Unsupportted opt method')
        if self.config.use_cost_model:
            self.tuner.buildCostModel(self.task)
        self.perf_buffer = perfBuffer(self.config)
        return task

    def evalSamples(self, samples):
        start = time.time()
        self.runner.run(samples, self.perf_buffer)
        end = time.time()
        print("Hardware time %f"%(end - start))
        return samples

def create_kwargs_ctx(task, option):
    builder = option["builder"]
    runner = option["runner"]
    attach_objects = runner.set_task(task)
    build_kwargs = runner.get_build_kwargs()
    builder.set_task(task, build_kwargs)
    def get():
        return build_kwargs
    get.n_parallel = builder.n_parallel
    get.attach_objects = attach_objects
    return get
