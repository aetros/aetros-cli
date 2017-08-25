from __future__ import division
from __future__ import absolute_import

import json
import os
import time
import psutil
from threading import Thread
import numpy as np
import signal


class MonitoringThread(Thread):
    def __init__(self, job_backend):
        Thread.__init__(self)

        self.job_backend = job_backend
        self.max_minutes = 0

        job = self.job_backend.job
        if 'maxTime' in job['config'] and job['config']['maxTime'] > 0:
            self.max_minutes = job['config']['maxTime']

        import sys
        if 'theano.sandbox' in sys.modules:
            # at this point, theano is already initialised, so we can use it to monitor the GPU.
            from theano.sandbox import cuda
            self.on_gpu = cuda.use.device_number is not None
        else:
            self.on_gpu = False

        self.stream = self.job_backend.git.stream_file('aetros/job/monitoring.csv')
        self.stream.write(json.dumps(["second", "cpu", "memory", "memory_gpu"])[1:-1] + "\n")
        self.second = 0
        self.started = time.time()
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

        minutes_run = (time.time() - self.started) / 60

        if not self.running:
            return

        if self.max_minutes > 0:
            if minutes_run > self.max_minutes:
                self.job_backend.logger.warning("\nMaxTime of %d/%d reached" % (minutes_run, self.max_minutes))
                self.job_backend.early_stop()

        gpu_memory_use = None

        import aetros.cuda_gpu
        info = aetros.cuda_gpu.get_memory(0)

        if info is not None:
            free, total = info
            gpu_memory_use = free/total*100

        self.stream.write(json.dumps([self.second, cpu_util, mem.percent, gpu_memory_use])[1:-1] + "\n")
        self.job_backend.git.store_file('aetros/job/times/elapsed.json', json.dumps(time.time() - self.started))

        self.second += 1
        pass