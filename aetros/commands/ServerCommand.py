from __future__ import absolute_import
from __future__ import print_function
import argparse
import os
import sys
import time
from threading import Lock

import paramiko as paramiko
import psutil
import subprocess

import signal

import requests
import six
from cryptography.hazmat.primitives import serialization
from requests.auth import HTTPBasicAuth

import aetros.api
from aetros.git import Git
from aetros.logger import GeneralLogger

from aetros.backend import EventListener, BackendClient
from aetros.utils import unpack_simple_job_id, read_home_config
import aetros.cuda_gpu

class ServerClient(BackendClient):
    def __init__(self, config, event_listener, logger):
        BackendClient.__init__(self, config, event_listener, logger)
        self.server_name = None
        self.go_offline_on_first_failed_attempt = False

    def configure(self, server_name):
        self.server_name = server_name

    def on_connect(self, reconnect=False):
        self.send_message({'type': 'register_server', 'server': self.server_name, 'reconnect': reconnect})
        messages = self.wait_for_at_least_one_message()

        if not messages:
            return False

        message = messages.pop(0)
        if isinstance(message, dict) and 'a' in message:
            if "registration_failed" in message['a']:
                self.logger.error('Failed. ' + message['reason'])
                self.close()
                self.event_listener.fire('failed')
                return False

            if "already_registered" in message['a']:
                self.logger.error("Registration failed. This server is already registered.")
                return False

            if "registered" in message['a']:
                self.registered = True
                self.event_listener.fire('registration', {'username': message['username'], 'server': self.server_name})
                self.handle_messages(messages)
                return True

        self.logger.error("Registration of server %s failed due to protocol error." % (self.server_name,))
        if message:
            self.logger.error("Got server response: " + str(message))

        return False

    def handle_messages(self, messages):
        for message in messages:
            if not isinstance(message, dict):
                return

            if 'stop' in message:
                self.close()
                self.event_listener.fire('stop')

            if 'type' in message:
                if message['type'] == 'jobs':
                    self.event_listener.fire('jobs', message['jobs'])


class ServerCommand:
    model = None
    job_model = None

    def __init__(self, logger):
        self.logger = logger
        self.last_utilization = None
        self.last_net = {}
        self.nets = []
        self.server = None
        self.ending = False
        self.active = True
        self.config = {}
        self.lock = Lock()

        self.general_logger_stdout = None
        self.general_logger_stderr = None

        self.executed_jobs = 0
        self.started_jobs = {}
        self.max_jobs = 0
        self.ssh_key_private = None
        self.ssh_key_public = None

        self.resources_limit = {}
        self.enabled_gpus = []

        self.job_processes = {}
        self.registered = False
        self.show_stdout = False

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' server')
        parser.add_argument('name', nargs='?', help="Server name")
        parser.add_argument('--generate-ssh-key', help="Generates automatically a ssh key, register them in AETROS in "
                                                       "your account, and delete them when the server exits. "
                                                       "You should prefer 'aetros register' command as its safer.")

        parser.add_argument('--allow-host-execution', action='store_true', help="Whether a job can run on this server "
            "directly, without a virtual (docker) container.\nSecurity risk and makes resource limitation useless.")

        parser.add_argument('--max-memory',
            help="How many RAM is available. In gigabyte. Per default all available memory.")
        parser.add_argument('--max-cpus',
            help="How many cores are available. Per default all available CPU cores.")
        parser.add_argument('--max-gpus',
            help="How many GPUs are available. Comma separate list of device ids (pciBusId)."
                 "Per default all available GPU cards. Use 'aetros gpu' too see the ids.")

        parser.add_argument('--no-gpus', action='store_true', help="Disable all GPUs")

        parser.add_argument('--max-jobs', help="How many jobs are allowed to run in total until the process exists automatically.")
        parser.add_argument('--host', help="Default trainer.aetros.com. Read from the global configuration ~/aetros.yml.")
        parser.add_argument('--show-stdout', action='store_true', help="Show all stdout of all jobs. Only for debugging necessary.")

        parsed_args = parser.parse_args(args)

        if not parsed_args.name:
            parser.print_help()
            sys.exit()

        self.config = read_home_config()

        if parsed_args.max_jobs:
            self.max_jobs = int(parsed_args.max_jobs)

        if parsed_args.max_memory:
            self.resources_limit['memory'] = int(parsed_args.max_memory)

        if parsed_args.max_cpus:
            self.resources_limit['cpus'] = int(parsed_args.max_cpus)

        self.resources_limit['host_execution'] = parsed_args.allow_host_execution

        gpus = []
        try:
            gpus = aetros.cuda_gpu.get_ordered_devices()
            for i in range(len(gpus)):
                self.enabled_gpus.append(i)
        except Exception: pass

        if parsed_args.max_gpus:
            self.enabled_gpus = []

            for i in parsed_args.max_gpus.split(','):
                i = int(i)
                if i < 0 or i >= len(gpus):
                    raise Exception('--max-gpus ' + str(i) + ' not available on the system. GPUs ' + str([i for i in range(len(gpus))])+ ' detected.')

                self.enabled_gpus.append(i)

        elif parsed_args.no_gpus:
            self.enabled_gpus = []

        if parsed_args.show_stdout:
            self.show_stdout = True

        event_listener = EventListener()

        event_listener.on('registration', self.registration_complete)
        event_listener.on('failed', self.connection_failed)
        event_listener.on('jobs', self.sync_jobs)
        event_listener.on('close', self.on_client_close)

        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, self.on_signusr1)

        ssh_key_registered = False
        if parsed_args.generate_ssh_key:
            self.logger.info('Generate SSH key')

            ssh_key = paramiko.RSAKey.generate(4096)
            self.ssh_key_private = ssh_key.key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()
            ).decode()
            self.ssh_key_public = 'rsa ' + ssh_key.get_base64() + ' ' + parsed_args.name

            self.logger.info('Register SSH key at ' + self.config['host'])

            data = {
                'name': parsed_args.name,
                'secure_key': parsed_args.generate_ssh_key,
                'key': self.ssh_key_public,
            }

            response = aetros.api.http_request('server/ssh-key', json_body=data, method='post')

            ssh_key_registered = response == True

        def delete_ssh_key():
            self.logger.info('Delete SSH key at ' + self.config['host'])

            data = {
                'secure_key': parsed_args.generate_ssh_key,
                'key': self.ssh_key_public,
            }
            response = aetros.api.http_request('server/ssh-key/delete', json_body=data)
            if not response:
                self.logger.error('Could not delete SSH key in AETROS Trainer.')

        if parsed_args.generate_ssh_key and ssh_key_registered:
            import atexit
            atexit.register(delete_ssh_key)

        if parsed_args.host:
            self.config['host'] = parsed_args.host

        if self.ssh_key_private:
            self.config['ssh_key_base64'] = self.ssh_key_private

        self.server = ServerClient(self.config, event_listener, self.logger)

        self.general_logger_stdout = GeneralLogger(job_backend=self, redirect_to=sys.__stdout__)
        self.general_logger_stderr = GeneralLogger(job_backend=self, redirect_to=sys.__stderr__)

        sys.stdout = self.general_logger_stdout
        sys.stderr = self.general_logger_stderr

        self.server.configure(parsed_args.name)
        self.logger.debug('Connecting to ' + self.config['host'])
        self.server.start()
        self.write_log("\n")

        try:
            while self.active:
                if self.registered:
                    self.server.send_message({'type': 'utilization', 'values': self.collect_system_utilization()})
                    self.check_finished_jobs()

                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.warning('Aborted')
            self.stop()

    def on_signusr1(self, signal, frame):
        self.logger.info("ending=%s, active=%s, registered=%s, %d running, %d messages, %d connection_tries" % (
            str(self.ending),
            str(self.active),
            str(self.registered),
            len(self.job_processes),
            len(self.server.queue),
            self.server.connection_tries,
        ))

        for full_id in six.iterkeys(self.job_processes):
            self.logger.info("Running " + full_id)

    def on_client_close(self, params):
        self.active = False
        self.logger.warning('Closed')

    def write_log(self, message):
        self.server.send_message({'type': 'log', 'message': message})
        return True

    def stop(self):
        self.active = False

        for p in six.itervalues(self.job_processes):
            p.kill()
            time.sleep(0.1)
            p.terminate()

        self.general_logger_stdout.flush()
        self.general_logger_stderr.flush()
        self.server.close()

    def end(self):
        self.ending = True

        for p in self.job_processes:
            p.wait()

        self.check_finished_jobs()
        self.stop()

    def connection_failed(self, params):
        self.active = False
        sys.exit(1)

    def sync_jobs(self, jobs):
        self.lock.acquire()

        # make sure we started all ids from "jobs".
        # if we have still active jobs not in jobs_ids, stop them
        for full_id, resources_assigned in six.iteritems(jobs):
            started_id = full_id + '-' + str(resources_assigned['time'])

            if started_id in self.started_jobs:
                # we got the same job id + timestamp twice, just ignore it
                continue

            self.started_jobs[started_id] = True
            self.execute_job(full_id, resources_assigned)

        self.lock.release()

    def check_finished_jobs(self):
        self.lock.acquire()

        delete_finished = []

        for full_job_id, process in six.iteritems(self.job_processes):
            exit_code = process.poll()
            model, job_id = unpack_simple_job_id(full_job_id)

            if exit_code is not None:
                # command ended
                if exit_code == 0:
                    self.logger.info('Finished job %s. Exit status: %s' % (full_job_id, str(exit_code)))
                if exit_code > 0:
                    reason = 'Failed job %s. Exit status: %s' % (full_job_id, str(exit_code))
                    self.logger.error(reason)

                self.server.send_message({'type': 'job-finished', 'id': full_job_id})
                delete_finished.append(full_job_id)

        for full_job_id in delete_finished:
            del self.job_processes[full_job_id]

        self.lock.release()

    def execute_job(self, full_id, resources_assigned):
        self.logger.info("Execute job %s ..." % (full_id, ))
        self.executed_jobs += 1

        with open(os.devnull, 'r+b', 0) as DEVNULL:
            my_env = os.environ.copy()
            my_env['AETROS_ATTY'] = '1'

            if self.ssh_key_private is not None:
                my_env['AETROS_SSH_KEY_BASE64'] = self.ssh_key_private

            args = [sys.executable, '-m', 'aetros', 'start']
            if resources_assigned['gpus']:
                for gpu_id in resources_assigned['gpus']:
                    args += ['--gpu-device', gpu_id]

            args += [full_id]
            self.logger.info('$ ' + ' '.join(args))
            self.server.send_message({'type': 'job-executed', 'id': full_id})

            # Since JobBackend sends SIGINT to its current process group, wit sends also to its parents when same pg.
            # We need to change the process group of the process, so this won't happen.
            # If we don't this, the process of ServerCommand receives the SIGINT as well.
            kwargs = {}
            if os.name == 'nt':
                kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs['preexec_fn'] = os.setsid

            process = subprocess.Popen(args, bufsize=1, env=my_env, stdin=DEVNULL,
                stderr=subprocess.PIPE, stdout=subprocess.PIPE, **kwargs)

            if self.show_stdout:
                self.general_logger_stdout.attach(process.stdout, read_line=True)
                self.general_logger_stderr.attach(process.stderr, read_line=True)

            self.job_processes[full_id] = process

    def registration_complete(self, params):
        self.registered = True
        self.logger.info("Server connected to %s as %s under account %s registered." % (self.config['host'], params['server'], params['username']))
        self.server.send_message({'type': 'system', 'values': self.collect_system_information()})

    def collect_system_information(self):
        values = {}
        mem = psutil.virtual_memory()
        values['memory_total'] = mem.total

        import cpuinfo
        cpu = cpuinfo.get_cpu_info()
        values['resources_limit'] = self.resources_limit
        values['cpu_name'] = cpu['brand']
        values['cpu'] = [cpu['hz_advertised_raw'][0], cpu['count']]
        values['nets'] = {}
        values['disks'] = {}
        values['gpus'] = {}
        values['boot_time'] = psutil.boot_time()

        try:
            for gpu_id, gpu in enumerate(aetros.cuda_gpu.get_ordered_devices()):
                gpu['available'] = gpu_id in self.enabled_gpus

                values['gpus'][gpu_id] = gpu
        except Exception: pass

        for disk in psutil.disk_partitions():
            try:
                name = self.get_disk_name(disk[1])
                values['disks'][name] = psutil.disk_usage(disk[1]).total
            except Exception:
                # suppress Operation not permitted
                pass

        try:
            for id, net in psutil.net_if_stats().items():
                if 0 != id.find('lo') and net.isup:
                    self.nets.append(id)
                    values['nets'][id] = net.speed or 1000
        except Exception:
            # suppress Operation not permitted
            pass

        return values

    def get_disk_name(self, name):

        if 0 == name.find("/Volumes"):
            return os.path.basename(name)

        return name

    def collect_system_utilization(self):
        values = {}

        values['cpu'] = psutil.cpu_percent(interval=0.2, percpu=True)
        mem = psutil.virtual_memory()
        values['memory'] = mem.percent
        values['disks'] = {}
        values['jobs'] = {'running': len(self.job_processes)}
        values['nets'] = {}
        values['processes'] = []
        values['gpus'] = {}

        try:
            for gpu_id, gpu in enumerate(aetros.cuda_gpu.get_ordered_devices()):
                values['gpus'][gpu_id] = aetros.cuda_gpu.get_memory(gpu['device'])
        except Exception: pass

        for disk in psutil.disk_partitions():
            try:
                name = self.get_disk_name(disk[1])
                values['disks'][name] = psutil.disk_usage(disk[1]).used
            except Exception: pass

        net_stats = psutil.net_io_counters(pernic=True)
        for id in self.nets:
            net = net_stats[id]
            values['nets'][id] = {
                'recv': net.bytes_recv,
                'sent': net.bytes_sent,
                'upload': 0,
                'download': 0
            }

            if id in self.last_net and self.last_utilization:
                values['nets'][id]['upload'] = (net.bytes_sent - self.last_net[id]['sent']) / (
                    time.time() - self.last_utilization)
                values['nets'][id]['download'] = (net.bytes_recv - self.last_net[id]['recv']) / (
                    time.time() - self.last_utilization)

            self.last_net[id] = dict(values['nets'][id])

        for p in psutil.process_iter():
            try:
                cpu = p.cpu_percent()
                if cpu > 1 or p.memory_percent() > 1:
                    values['processes'].append([
                        p.pid,
                        p.name(),
                        p.username(),
                        p.create_time(),
                        p.status(),
                        p.num_threads(),
                        p.memory_percent(),
                        cpu
                    ])
            except OSError:
                pass
            except psutil.Error:
                pass

        try:
            if hasattr(os, 'getloadavg'):
                values['loadavg'] = os.getloadavg()
            else:
                values['loadavg'] = ''
        except OSError:
            pass

        self.last_utilization = time.time()
        return values
