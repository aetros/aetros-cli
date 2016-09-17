from __future__ import division
from __future__ import absolute_import
import time
import psutil
from threading import Thread
import numpy as np


class MonitoringThread(Thread):
    def __init__(self, aetros_backend, trainer):
        Thread.__init__(self)
        self.trainer = trainer
        self.aetros_backend = aetros_backend
        self.second = 0
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            self.monitor()
            time.sleep(0.1)

    def monitor(self):
        cpu_util = np.mean(psutil.cpu_percent(interval=1, percpu=True))
        mem = psutil.virtual_memory()

        gpu_memory_use = None
        import keras.backend as K

        if self.trainer.on_gpu and K.backend() == 'theano':
            from theano.sandbox import cuda
            if cuda.cuda_ndarray.cuda_ndarray.mem_info:
                gpu = cuda.cuda_ndarray.cuda_ndarray.mem_info()

                if gpu[1] != 0:
                    #sometimes, theano returns 0 as total memory, which will fail in next line
                    gpu_memory_use = (gpu[1] - gpu[0]) / gpu[1] * 100

        self.aetros_backend.job_add_status('system', {
            'second': self.second,
            'cpu': cpu_util,
            'memory': mem.percent,
            'memory_gpu': gpu_memory_use
        })

        self.second += 1
        pass
