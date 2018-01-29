from __future__ import division
from __future__ import absolute_import

import simplejson
import random
import time
import math

import docker
import docker.utils
import docker.errors
import psutil

from threading import Thread

import six

import aetros.cuda_gpu
import numpy as np


class MonitoringThread(Thread):
    def __init__(self, job_backend, cpu_cores=1, gpu_devices=None, docker_container=None):
        Thread.__init__(self)

        self.job_backend = job_backend
        self.gpu_devices = gpu_devices
        self.docker_container = docker_container
        self.max_minutes = 0
        self.cpu_cores = cpu_cores

        job = self.job_backend.job
        if 'maxTime' in job['config'] and isinstance(job['config']['maxTime'], int) and job['config']['maxTime'] > 0:
            self.max_minutes = job['config']['maxTime']

        self.hardware_stream = self.job_backend.git.stream_file('aetros/job/monitoring.csv')

        header = ["second", "cpu", "memory"]
        try:
            if self.gpu_devices:
                for gpu_id, gpu in enumerate(aetros.cuda_gpu.get_ordered_devices()):
                    if gpu_id in gpu_devices:
                        header.append("memory_gpu" + str(gpu['id']))
        except aetros.cuda_gpu.CudaNotImplementedException: pass

        if job_backend.get_job_model().has_dpu():
            header += ['dpu0']

        self.hardware_stream.write(simplejson.dumps(header)[1:-1] + "\n")
        self.running = True
        self.early_stopped = False
        self.handle_max_time = True
        self.client = docker.from_env()
        self.docker_api = docker.APIClient(**docker.utils.kwargs_from_env())
        self.stat_stream = None

        self.docker_last_last_reponse = None
        self.docker_last_stream_data = 0
        self.docker_last_mem = None
        self.docker_last_cpu = None

    def stop(self):
        self.running = False

    def run(self):
        def docker_stats_reader(response):
            previous_cpu = 0
            previous_system = 0

            stream = self.docker_api._stream_helper(response)

            try:
                for line in stream:
                    data = simplejson.loads(line)
                    if 'cpu_stats' not in data or not data['cpu_stats']:
                        return

                    if 'system_cpu_usage' not in data['cpu_stats']:
                        return

                    cpu_util = 0
                    cpu_delta = data['cpu_stats']['cpu_usage']['total_usage'] - previous_cpu
                    system_delta = data['cpu_stats']['system_cpu_usage'] - previous_system

                    previous_cpu = data['cpu_stats']['cpu_usage']['total_usage']
                    previous_system = data['cpu_stats']['system_cpu_usage']

                    if cpu_delta > 0 and system_delta > 0:
                        cpu_cores = len(data['cpu_stats']['cpu_usage']['percpu_usage'])

                        cpu_util = (cpu_delta / system_delta) * cpu_cores / self.cpu_cores * 100

                    mem_util = data['memory_stats']['usage'] / data['memory_stats']['limit'] * 100
                    self.docker_last_stream_data = time.time()
                    self.docker_last_cpu = min(cpu_util, 100)
                    self.docker_last_mem = min(mem_util, 100)
            except Exception:
                return

        docker_reader = None

        while self.running:
            self.handle_early_stop()

            self.job_backend.git.store_file('aetros/job/times/elapsed.json', simplejson.dumps(time.time() - self.job_backend.start_time))

            if self.job_backend.is_paused:
                # when paused, we do not monitor anything, except elapsed.
                time.sleep(1)
                continue

            # always sent network information even when marked as ended. The real end will tear down this thread.
            self.network_sync()

            if self.job_backend.ended:
                # stop hardware monitoring when ended
                time.sleep(1)
                continue

            if self.docker_container:
                if docker_reader and self.docker_last_last_reponse and time.time()-self.docker_last_stream_data > 3:
                    self.docker_last_last_reponse.close()
                    docker_reader.join()

                if not docker_reader or not docker_reader.isAlive():
                    url = self.docker_api._url("/containers/{0}/stats", self.docker_container)
                    self.docker_last_last_reponse = self.docker_api._get(url, stream=True)

                    docker_reader = Thread(target=docker_stats_reader, args=[self.docker_last_last_reponse])
                    docker_reader.daemon = True
                    docker_reader.start()

                if self.docker_last_cpu is not None:
                    self.monitor(self.docker_last_cpu, self.docker_last_mem)

                time.sleep(1)
            else:
                cpu_util = np.mean(psutil.cpu_percent(interval=1, percpu=True))  # blocks 1sec
                mem_util = psutil.virtual_memory().percent
                self.monitor(cpu_util, mem_util) #takes always at least 1sec, no need for sleep
                time.sleep(0.01)

        # thread requested to end. So queue last network sync update
        self.network_sync()

    def handle_early_stop(self):
        if not self.early_stopped and self.handle_max_time and self.max_minutes > 0:
            minutes_run = (time.time() - self.job_backend.start_time) / 60
            if minutes_run > self.max_minutes:
                self.early_stopped = True
                self.job_backend.logger.warning("Max time of "+str(self.max_minutes)+" minutes reached.")
                self.job_backend.early_stop()

    def network_sync(self):
        if self.job_backend.client.write_speeds:
            network = {
                'ended': self.job_backend.ended,
                'channels': {},
                'messages': 0,
                'files': {},
                'git': [],
                'sent': self.job_backend.client.bytes_sent,
                'total': self.job_backend.client.bytes_total,
                'speed': self.job_backend.client.bytes_speed,
            }

            for channel, messages in six.iteritems(self.job_backend.client.queues):
                messages = messages[:]

                for message in messages:
                    if 'type' not in message:
                        continue

                    if message['type'] == 'store-blob' and message['path'] in ['aetros/job/network.json']:
                        continue

                    if message['type'] == 'git-unpack-objects':
                        bytes_sent = message['_bytes_sent']
                        total = message['_total']
                        network['git'].append({
                            'sent': bytes_sent,
                            'total': total,
                            'objects': message['objects']
                        })
                        network['messages'] += 1

                    if message['type'] == 'store-blob':
                        if message['path'].startswith('aetros/job/times'):
                            continue
                        bytes_sent = message['_bytes_sent']
                        total = message['_total']
                        network['files'][message['path']] = {
                            'sent': bytes_sent,
                            'total': total,
                        }

                        network['messages'] += 1

            data = simplejson.dumps(network)
            self.job_backend.client.send({'type': 'store-blob', 'path': 'aetros/job/network.json', 'data': data}, channel='')

    def monitor(self, cpu_util, mem_util):
        x = math.ceil(time.time()-self.job_backend.start_time)

        row = [x, cpu_util, mem_util]

        try:
            if self.gpu_devices:
                for gpu_id, gpu in enumerate(aetros.cuda_gpu.get_ordered_devices()):
                    if gpu_id not in self.gpu_devices:
                        continue

                    gpu_memory_use = None
                    info = aetros.cuda_gpu.get_memory(gpu['device'])

                    if info is not None:
                        free, total = info
                        gpu_memory_use = (total-free) / total*100

                    row.append(gpu_memory_use)
        except aetros.cuda_gpu.CudaNotImplementedException: pass

        if self.job_backend.get_job_model().has_dpu():
            row += [25 + random.randint(-10, 20)]

        self.hardware_stream.write(simplejson.dumps(row)[1:-1] + "\n")
