from __future__ import division
from __future__ import absolute_import

import json
import time
import psutil
from threading import Thread
import numpy as np
import six
import aetros.cuda_gpu


class MonitoringThread(Thread):
    def __init__(self, job_backend, start_time=None):
        Thread.__init__(self)

        self.job_backend = job_backend
        self.max_minutes = 0

        job = self.job_backend.job
        if 'maxTime' in job['config'] and job['config']['maxTime'] > 0:
            self.max_minutes = job['config']['maxTime']

        self.stream = self.job_backend.git.stream_file('aetros/job/monitoring.csv')

        header = ["second", "cpu", "memory"]
        for gpu_id in six.iterkeys(aetros.cuda_gpu.get_ordered_devices()):
            header.append("memory_gpu" + str(gpu_id))

        self.stream.write(json.dumps(header)[1:-1] + "\n")
        self.second = 0
        self.started = start_time or time.time()
        self.running = True
        self.early_stopped = False
        self.handle_max_time = True
        self.handle_max_time_time = self.started

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            self.monitor()
            time.sleep(0.01)

    def monitor(self):
        if self.early_stopped:
            return

        cpu_util = np.mean(psutil.cpu_percent(interval=1, percpu=True)) #blocks 1sec
        mem = psutil.virtual_memory()


        if not self.running:
            return

        if self.handle_max_time and self.max_minutes > 0:
            minutes_run = (time.time() - self.handle_max_time_time) / 60
            if minutes_run > self.max_minutes:
                self.early_stopped = True
                self.job_backend.logger.warning("Max time of "+str(self.max_minutes)+" minutes reached.")
                self.job_backend.early_stop()

        row = [self.second, cpu_util, mem.percent]

        for gpu in six.itervalues(aetros.cuda_gpu.get_ordered_devices()):
            gpu_memory_use = None
            info = aetros.cuda_gpu.get_memory(gpu['device'])

            if info is not None:
                free, total = info
                gpu_memory_use = free/total*100

            row.append(gpu_memory_use)

        self.stream.write(json.dumps(row)[1:-1] + "\n")
        self.job_backend.git.store_file('aetros/job/times/elapsed.json', json.dumps(time.time() - self.started))

        self.second += 1
        pass