from __future__ import absolute_import
from __future__ import print_function
import argparse
import json
import os
import sys
import time
import psutil
import subprocess

import signal

import requests
import six
from requests.auth import HTTPBasicAuth

from aetros.api import raise_response_exception
from aetros.git import Git
from aetros.logger import GeneralLogger

from aetros.backend import EventListener, BackendClient
from aetros.utils import unpack_simple_job_id, read_home_config
import aetros.cuda_gpu

class ServerClient(BackendClient):
    def __init__(self, host, event_listener, logger):
        BackendClient.__init__(self, host, event_listener, logger)
        self.server_name = None

    def configure(self, server_name):
        self.server_name = server_name

    def on_connect(self, reconnect=False):
        self.send_message({'type': 'register_server', 'server': self.server_name})
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
                if message['type'] == 'queue-jobs':
                    self.event_listener.fire('queue-jobs', message['jobs'])

                if message['type'] == 'unqueue-jobs':
                    self.event_listener.fire('unqueue-jobs', message['jobs'])

                if message['type'] == 'stop-job':
                    self.event_listener.fire('stop-job', message['id'])


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
        self.queue = {}
        self.queuedMap = {}

        self.general_logger_stdout = None
        self.general_logger_stderr = None

        self.executed_jobs = 0
        self.max_jobs = 0
        self.ssh_key_path = None

        self.resources_limit = {}
        self.disabled_gpus = []

        self.job_processes = []
        self.registered = False
        self.show_stdout = False

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' server')
        parser.add_argument('name', nargs='?', help="Server name")
        parser.add_argument('--generate-ssh-key', help="Generates automatically a ssh key, register them in AETROS in your account, and delete them when the server exits.")

        parser.add_argument('--allow-host-execution', action='store_true', help="Whether a job can run on this server "
            "directly, without a virtual (docker) container.\nSecurity risk and makes resource limitation useless.")

        parser.add_argument('--max-memory',
            help="How many RAM is available. In gigabyte. Per default all available memory.")
        parser.add_argument('--max-cpus',
            help="How many cores are available. Per default all available CPU cores.")
        parser.add_argument('--max-gpus',
            help="How many GPUs are available. Comma separate list of device ids. Starting at 0. "
                 "Per default all available GPU cards.")

        parser.add_argument('--no-gpus', action='store_true', help="Disable all GPUs")

        parser.add_argument('--max-jobs', help="How many jobs are allowed to run in total until the process exists automatically.")
        parser.add_argument('--host', help="Default trainer.aetros.com. Read from the global configuration ~/.aetros.yml.")
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

        if parsed_args.max_gpus or parsed_args.no_gpus:
            gpus = 0
            try:
                gpus = aetros.cuda_gpu.get_installed_devices()
            except:
                pass

            if parsed_args.max_gpus:
                for i in parsed_args.max_gpus.split(','):
                    if i > gpus or i < 0:
                        raise Exception('--resources-max-gpus ' + str(i) + ' not available on the system. Max ' + str(gpus)+ ' GPUs detected.')

                    self.disabled_gpus.append(i)

            elif parsed_args.no_gpus:
                for i in range(0, gpus):
                    self.disabled_gpus.append(i)

        if parsed_args.show_stdout:
            self.show_stdout = True

        event_listener = EventListener()

        event_listener.on('registration', self.registration_complete)
        event_listener.on('failed', self.connection_failed)
        event_listener.on('queue-jobs', self.queue_jobs)
        event_listener.on('unqueue-jobs', self.unqueue_jobs)
        event_listener.on('queue-ok', self.queue_ok)
        event_listener.on('stop-job', self.stop_job)
        event_listener.on('close', self.on_client_close)

        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, self.on_signusr1)

        ssh_key_registered = False
        if parsed_args.generate_ssh_key:

            self.ssh_key_path = os.path.normpath(os.path.expanduser('~/.ssh/id_' + parsed_args.name.replace('/', '__') + '_rsa'))
            if not os.path.exists(self.ssh_key_path):
                self.logger.info('Generate SSH key')
                subprocess.check_output(['ssh-keygen', '-q', '-N', '', '-t', 'rsa', '-b', '4048', '-f', self.ssh_key_path])

            self.logger.info('Register SSH key at ' + self.config['host'])
            url = 'https://' + self.config['host'] + '/api/server/ssh-key'

            with open(self.ssh_key_path +'.pub', 'r') as f:
                data = {
                    'name': parsed_args.name,
                    'secure_key': parsed_args.generate_ssh_key,
                    'key': f.read(),
                }

                auth = None
                if 'auth_user' in self.config:
                    auth = HTTPBasicAuth(self.config['auth_user'], self.config['auth_pw'])

                response = requests.post(url, data, auth=auth, verify=self.config['ssl_verify'], headers={'Accept': 'application/json'})

                if response.status_code != 200:
                    raise_response_exception('Could not register SSH key in AETROS Trainer.', response)

                ssh_key_registered = response.content == 'true'

        def delete_ssh_key():
            with open(self.ssh_key_path +'.pub', 'r') as f:
                data = {
                    'secure_key': parsed_args.generate_ssh_key,
                    'key': f.read(),
                }
                self.logger.info('Delete SSH key at ' + self.config['host'])
                url = 'https://' + self.config['host'] + '/api/server/ssh-key/delete'

                auth = None
                if 'auth_user' in self.config:
                    auth = HTTPBasicAuth(self.config['auth_user'], self.config['auth_pw'])

                response = requests.post(url, data, auth=auth, verify=self.config['ssl_verify'], headers={'Accept': 'application/json'})

                if response.status_code != 200:
                    raise_response_exception('Could not delete SSH key in AETROS Trainer.', response)

                os.unlink(self.ssh_key_path)
                os.unlink(self.ssh_key_path +'.pub')

        if parsed_args.generate_ssh_key and ssh_key_registered:
            import atexit
            atexit.register(delete_ssh_key)

        if parsed_args.host:
            self.config['host'] = parsed_args.host

        if self.ssh_key_path:
            self.config['ssh_key'] = self.ssh_key_path

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
                    self.process_queue()

                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.warning('Aborted')
            self.stop()

    def on_signusr1(self, signal, frame):
        self.logger.info("ending=%s, active=%s, %d queued, %d running" % (
            str(self.ending),
            str(self.active),
            self.queued_count(),
            len(self.job_processes)
        ))

    def on_client_close(self, params):
        self.active = False
        self.logger.warning('Closed')

    def write_log(self, message):
        self.server.send_message({'type': 'log', 'message': message})
        return True

    def stop(self):
        self.active = False

        for p in self.job_processes:
            p.kill()

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

    def stop_job(self, id):
        if id in self.queuedMap:
            job = self.queuedMap[id]
            self.logger.info("Queued job removed %s (priority: %s) " % (job['id'], job['priority']))

            # removing from the queue is enough, since the job process itself terminates it when job is aborted.
            if job in self.queue:
                self.queue[job['priority']].remove(job)

            del self.queuedMap[id]

    def unqueue_jobs(self, jobs):
        for id in jobs:
            if id in self.queuedMap:
                self.logger.info('Removed job %s from queue.' % (id, ))

                for priority in self.queue:
                    if self.queuedMap[id] in self.queue[priority]:
                        self.queue[priority].remove(self.queuedMap[id])

                del self.queuedMap[id]

    def queue_jobs(self, jobs):
        self.logger.debug('Got queue list with %d items.' % (len(jobs), ))

        for id in jobs.keys():
            self.check_finished_jobs()

            job = jobs[id]
            priority = job['priority']
            job['id'] = id

            if self.is_job_queued(id):
                self.logger.debug("Requested job %s is already known. Exclude that one." % (id, ))
                return

            self.logger.info("Queued job %s (priority:%d) in %s ..." % (
               job['id'], job['priority'],os.getcwd()
            ))

            self.queuedMap[job['id']] = job

            # add the job into the wait list
            if job['priority'] not in self.queue:
                self.queue[priority] = []

            self.queue[priority].append(job)

    def is_job_queued(self, id):
        return id in self.queuedMap

    def queued_count(self):
        i = 0
        for jobs in six.itervalues(self.queue):
            i += len(jobs)

        return i

    def is_job_running(self, id):
        for process in self.job_processes:
            job = getattr(process, 'job')
            if job['id'] == id:
                return True

        return False

    def queue_ok(self, id):
        """
        We queued the job, told the server so and server said we're ok to start the job now.

        :param id: 
        :return: 
        """
        job = self.queuedMap[id]
        priority = job['priority']

        self.logger.debug("Queued job confirmed %s (priority: %s) " % (job['id'], priority))

        # add the job into the wait list
        if job['priority'] not in self.queue:
            self.queue[priority] = []

        self.queue[priority].append(job)

    def check_finished_jobs(self):
        for process in self.job_processes:
            job = getattr(process, 'job')
            exit_code = process.poll()
            model, job_id = unpack_simple_job_id(job['id'])

            if exit_code is not None:
                # command ended

                if exit_code == 0:
                    self.logger.info('Finished job %s. Exit status: %s' % (job['id'], str(exit_code)))
                if exit_code > 0:
                    reason = 'Failed job %s. Exit status: %s' % (job['id'], str(exit_code))
                    self.logger.error(reason)

                if job['id'] in self.queuedMap:
                    del self.queuedMap[job['id']]

                git = Git(self.logger, None, self.config, model)
                if git.is_job_fetched(job_id):
                    git.read_job(job_id)

                    if not git.has_file('aetros/job/log.txt'):
                        log = six.b('')
                        if id(process.stdout) in self.general_logger_stdout.attach_last_messages:
                            log += self.general_logger_stdout.attach_last_messages[id(process.stdout)]

                        if id(process.stderr) in self.general_logger_stderr.attach_last_messages:
                            log += self.general_logger_stderr.attach_last_messages[id(process.stderr)]

                        git.commit_file('LOGS', 'aetros/job/log.txt', log)

                    from aetros.const import JOB_STATUS
                    if exit_code > 0 or not git.has_file('aetros/job/status/progress.json'):
                        self.logger.info("Set progress to " + ('CRASHED' if exit_code > 0 else 'DONE'))
                        git.commit_file(
                            'STOP',
                            'aetros/job/status/progress.json',
                            str(JOB_STATUS.PROGRESS_STATUS_CRASHED) if exit_code > 0 else str(JOB_STATUS.PROGRESS_STATUS_DONE)
                        )

                    git.push()
                    git.clean_up()

                self.server.send_message({'type': 'job-finished', 'id': job['id']})

        # remove dead job processes
        self.job_processes = [x for x in self.job_processes if x.poll() is None]

    def process_queue(self):
        self.check_finished_jobs()

        if self.ending:
            return

        if self.max_jobs and self.executed_jobs >= self.max_jobs:
            self.logger.warning('Limit of max jobs %d/%d reached. Waiting for active jobs to finish ...' % (self.executed_jobs, self.max_jobs))
            self.end()
            return

        # sort by priority: The higher the sooner the job starts
        for priority in sorted(self.queue, reverse=True):
            q = self.queue[priority]

            if len(q) > 0:
                # registered and free space for new jobs, so execute another one
                self.execute_job(q.pop(0))
                break

    def execute_job(self, job):
        self.logger.info("Execute job %s (priority=%s) in %s ..." % (job['id'], job['priority'], os.getcwd()))

        self.executed_jobs += 1

        with open(os.devnull, 'r+b', 0) as DEVNULL:
            my_env = os.environ.copy()
            my_env['AETROS_ATTY'] = '1'

            if self.ssh_key_path is not None:
                my_env['AETROS_SSH_KEY'] = self.ssh_key_path

            args = [sys.executable, '-m', 'aetros', 'start', job['id']]
            self.logger.info('$ ' + ' '.join(args))
            self.server.send_message({'type': 'job-executed', 'id': job['id']})

            # Since JobBackend sends SIGINT to its current process group to send the signal also to all its children
            # we need to change the process group of the process.
            # If we don't this, the process of ServerCommand receives the SIGINT as well.
            kwargs = {}
            if os.name == 'nt':
                kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs['preexec_fn'] = os.setsid

            process = subprocess.Popen(args, bufsize=1, env=my_env, stdin=DEVNULL,
                stderr=subprocess.PIPE, stdout=subprocess.PIPE, **kwargs)

            if self.show_stdout:
                self.general_logger_stdout.attach(process.stdout)
                self.general_logger_stderr.attach(process.stderr)

            setattr(process, 'job', job)
            self.job_processes.append(process)

    def registration_complete(self, params):
        self.registered = True

        # upon registration, we need to clear the queue, since the server sends us immediately
        # all to be enqueued jobs after registration/re-connection
        self.queue = {}
        self.queueMap = {}

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
        values['gpus'] = []
        values['boot_time'] = psutil.boot_time()

        try:
            for i in range(0, aetros.cuda_gpu.get_installed_devices()):
                gpu = aetros.cuda_gpu.get_device_properties(i)
                gpu['available'] = i not in self.disabled_gpus
                values['gpus'].append(gpu)
        except:
            pass

        for disk in psutil.disk_partitions():
            try:
                name = self.get_disk_name(disk[1])
                values['disks'][name] = psutil.disk_usage(disk[1]).total
            except:
                # suppress Operation not permitted
                pass

        try:
            for id, net in psutil.net_if_stats().items():
                if 0 != id.find('lo') and net.isup:
                    self.nets.append(id)
                    values['nets'][id] = net.speed or 1000
        except:
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
        values['jobs'] = {'enqueued': self.queued_count(), 'running': len(self.job_processes)}
        values['nets'] = {}
        values['processes'] = []
        values['gpus'] = []

        try:
            for i in range(0, aetros.cuda_gpu.get_installed_devices()):
                values['gpus'].append(aetros.cuda_gpu.get_memory(i))
        except:
            pass

        for disk in psutil.disk_partitions():
            try:
                name = self.get_disk_name(disk[1])
                values['disks'][name] = psutil.disk_usage(disk[1]).used
            except:
                pass

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
