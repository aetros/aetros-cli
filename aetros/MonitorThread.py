from __future__ import division
from __future__ import absolute_import
import time
import psutil
from threading import Thread
import numpy as np

class MonitoringThread(Thread):
    def __init__(self, job_backend):
        Thread.__init__(self)

        import sys
        if 'theano.sandbox' in sys.modules:
            # at this point, theano is already initialised, so we can use it to monitor the GPU.
            from theano.sandbox import cuda
            self.on_gpu = cuda.use.device_number is not None
        else:
            self.on_gpu = False

        self.job_backend = job_backend
        self.second = 0
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            self.monitor()
            time.sleep(0.05)

    def monitor(self):
        cpu_util = np.mean(psutil.cpu_percent(interval=1, percpu=True))
        mem = psutil.virtual_memory()

        gpu_memory_use = None

        if self.on_gpu:
            from theano.sandbox import cuda
            if cuda.cuda_ndarray.cuda_ndarray.mem_info:
                gpu = cuda.cuda_ndarray.cuda_ndarray.mem_info()

                if gpu[1] != 0:
                    #sometimes, theano returns 0 as total memory, which will fail in next line
                    gpu_memory_use = (gpu[1] - gpu[0]) / gpu[1] * 100

        self.job_backend.job_add_status('system', {
            'second': self.second,
            'cpu': cpu_util,
            'memory': mem.percent,
            'memory_gpu': gpu_memory_use
        })

        self.second += 1
        pass
