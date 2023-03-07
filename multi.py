import multiprocessing as _multi
#multi = _multi.get_context("spawn")
multi = _multi.get_context("fork")
import os
import time
import psutil
import signal
import math

class Job:
    def __init__(self, func, attach_info, timeout = 10):
        self.func = func
        self.attach_info = attach_info
        self.timeout = timeout

    def start(self, inputs):
        self.queue = multi.Queue(2)
        self.process = multi.Process(target=exec, args = (self.func, self.queue, inputs,))
        self.process.start()

    def get(self):
        try:
            res = self.queue.get(block=True, timeout=self.timeout)
        except:
            res = 0
        if self.process.is_alive():
            kill_child_processes(self.process.pid)
            self.process.terminate()
        self.process.join()
        self.queue.close()
        self.queue.join_thread()
        del self.process; del self.queue
        return res

def exec(func, queue, args):
    res = func(*args)
    queue.put(res)

def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        try:
            process.send_signal(sig)
        except psutil.NoSuchProcess:
            return

