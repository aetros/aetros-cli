from __future__ import absolute_import
from __future__ import print_function
import argparse
import json
import os
import pprint
import socket
import sys
from threading import Lock, Thread
import time

import select

import psutil
import subprocess

from aetros.backend import invalid_json_values, EventListener, parse_message


class Server:
    """
    :type: event_listener : EventListener
    """

    def __init__(self, api_host, server_api_token, event_listener):
        self.event_listener = event_listener
        self.server_api_token = server_api_token
        self.api_host = api_host
        self.active = False
        self.connected = False
        self.registered = False
        self.read_buffer = ''
        self.s = None
        self.thread_instance = None

        self.api_port = int(os.getenv('API_PORT') or 8051)

        if not self.api_host:
            self.api_host = os.getenv('API_HOST')
            if not self.api_host or self.api_host == 'false':
                self.api_host = 'aetros.com'

        self.lock = Lock()

    def start(self):
        self.active = True

        if not self.thread_instance:
            self.thread_instance = Thread(target=self.thread)
            self.thread_instance.daemon = True
            self.thread_instance.start()

    def connect(self):
        locked = False

        try:
            self.lock.acquire()
            locked = True
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((self.api_host, self.api_port))
            self.connected = True
            self.lock.release()
            locked = False

            self.send_message({'register_server': self.server_api_token})
            messages = self.read_full_message(self.s)

            if isinstance(messages, list):
                if "ACCESS_DENIED" in messages:
                    print('Access denied')
                    return False

                if "SERVER_ALREADY_REGISTERED" in messages:
                    print("Registration failed. This server is already registered.")
                    return

                if "SERVER_REGISTERED" in messages:
                    self.registered = True
                    for message in messages:
                        if 'registered' in message:
                            server_id = message['id']
                            self.event_listener.fire('registration', server_id)
                            print("Connected to %s as server %s" % (self.api_host, server_id))
                    self.handle_messages(messages)
                    return True

            print("Registration failed due to protocol error.")
            return False
        except socket.error as error:
            if locked:
                self.lock.release()
            print("Connection error during connecting to %s: %d: %s." % (self.api_host, error.errno, error.message))

            return False

    def thread(self):
        last_ping = 0

        while True:
            if self.connected and self.registered:
                try:
                    if last_ping < time.time() - 10:
                        # ping every second
                        last_ping = time.time()
                        self.send_message("PING")

                    if self.connected:
                        # see if we can read something
                        self.lock.acquire()
                        readable, writable, exceptional = select.select([self.s], [self.s], [])
                        self.lock.release()
                        if exceptional:
                            self.connection_error()

                        for s in readable:
                            messages = self.read(s)
                            if messages:
                                self.handle_messages(messages)

                except socket.error as error:
                    self.connection_error(error)

            elif not self.connected and self.active:
                if not self.connect():
                    time.sleep(5)

            time.sleep(0.1)

    def close(self):
        self.active = False
        self.connected = False

        self.lock.acquire()
        try:
            self.s.shutdown(socket.SHUT_RDWR)
            self.s.close()
        except:
            pass
        self.lock.release()

    def handle_messages(self, messages):
        for message in messages:

            if not isinstance(message, dict):
                continue

            if 'stop' in message:
                self.close()
                self.event_listener.fire('stop')

            if 'type' in message:
                if message['type'] == 'start-jobs':
                    self.event_listener.fire('start-jobs', message['jobs'])

    def connection_error(self, error=None):
        if error:
            print("Connection error: %d: %s" % (error.errno, error.message,))
        else:
            print("Connection error")

        self.connected = False
        self.registered = False

    def send_message(self, message):
        sent = False
        if isinstance(message, dict):
            message['sending'] = True

        msg = json.dumps(message, default=invalid_json_values)
        self.lock.acquire()

        try:
            self.s.sendall(msg + "\n")
        except:
            sent = False
            self.connected = False

        self.lock.release()

        return sent

    def read_full_message(self, s):
        """
        Reads until we receive a message termination (\n)
        """
        buffer = ''

        while True:
            chunk = ''
            try:
                self.lock.acquire()
                chunk = s.recv(2048)
            finally:
                self.lock.release()

            if chunk == '':
                self.connection_error()
                return False

            buffer += chunk

            buffer, parsed = parse_message(buffer)
            self.read_buffer = buffer
            if parsed:
                return parsed

    def read(self, s):
        """
        Reads per call current buffer from network stack. If a full message has been collected (\n retrieved)
        the message will be parsed and returned. If no message has yet been completley transmitted it returns []

        :return: list
        """

        chunk = ''

        try:
            self.lock.acquire()
            chunk = s.recv(2048)
        finally:
            self.lock.release()

        if chunk == '':
            self.connection_error()
            return False

        self.read_buffer += chunk

        self.read_buffer, parsed = parse_message(self.read_buffer)
        return parsed


class ServerCommand:
    model = None
    job_model = None

    def __init__(self):
        self.last_utilization = None
        self.last_net = {}
        self.nets = []
        self.server = None
        self.queue = []
        self.jobs = []
        self.max_parallel_jobs = 2
        self.registered = False

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' server')
        parser.add_argument('--secure-key', nargs='?',
                            help="Secure key of the server.")
        parser.add_argument('--max-jobs', help="How many jobs should be run at the same time.")
        parser.add_argument('--server', help="Default aetros.com.")

        parsed_args = parser.parse_args(args)

        if not parsed_args.secure_key:
            parser.print_help()
            sys.exit()

        if parsed_args.max_jobs:
            self.max_parallel_jobs = int(parsed_args.max_jobs)

        event_listener = EventListener()

        event_listener.on('registration', self.registration_complete)
        event_listener.on('start-jobs', self.start_jobs)

        self.server = Server(parsed_args.server, parsed_args.secure_key, event_listener)
        self.server.start()

        while True:
            if self.registered:
                self.server.send_message({'type': 'utilization', 'values': self.collect_system_utilization()})
                self.process_queue()

            time.sleep(1)

    def start_jobs(self, jobs):
        for job in jobs:
            self.start_job(job)

    def start_job(self, job):
        print("Queued job %s#%d (%s) by %s in %s ..." % (job['modelId'], job['index'], job['id'], job['username'], os.getcwd()))

        self.server.send_message({'type': 'job-queued', 'id': job['id']})

        self.queue.append(job)

    def process_queue(self):
        # reject failed commands
        failed_jobs = [x for x in self.jobs if x.poll() > 0]

        for failed_process in failed_jobs:
            reason = 'Exit status: ' + failed_process.poll()
            self.server.send_message({'type': 'job-failed', 'job': getattr(failed_process, 'job'), 'reason': reason})

        # remove dead job processes
        self.jobs = [x for x in self.jobs if x.poll() is None]

        if len(self.jobs) >= self.max_parallel_jobs:
            return

        if len(self.queue) > 0:
            # registered and free space for new jobs, so execute another one
            self.execute_job(self.queue.pop(0))

    def execute_job(self, job):
        print("Execute job %s#%d (%s) by %s in %s ..." % (job['modelId'], job['index'], job['id'], job['username'], os.getcwd()))

        with open(os.devnull, 'r+b', 0) as DEVNULL:
            my_env = os.environ.copy()

            if 'PYTHONPATH' not in my_env:
                my_env['PYTHONPATH'] = ''

            my_env['PYTHONPATH'] += ':' + os.getcwd()
            args = [sys.executable, '-m', 'aetros', 'start', job['id'], '--secure-key=' + job['apiKey']]
            process = subprocess.Popen(args, stdin=DEVNULL, stdout=sys.stdout, stderr=sys.stderr, close_fds=True, env=my_env)
            setattr(process, 'job', job)
            self.jobs.append(process)

    def registration_complete(self, params):
        self.registered = True
        self.server.send_message({'type': 'system', 'values': self.collect_system_information()})

    def collect_system_information(self):
        values = {}
        mem = psutil.virtual_memory()
        values['memory_total'] = mem.total

        import cpuinfo
        cpu = cpuinfo.get_cpu_info()
        values['cpu_name'] = cpu['brand']
        values['cpu'] = [cpu['hz_actual_raw'][0], cpu['count']]
        values['nets'] = {}
        values['disks'] = {}
        values['boot_time'] = psutil.boot_time()

        for disk in psutil.disk_partitions():
            name = self.get_disk_name(disk[1])
            values['disks'][name] = psutil.disk_usage(disk[1]).total

        for id, net in psutil.net_if_stats().iteritems():
            if 0 != id.find('lo') and net.isup:
                self.nets.append(id)
                values['nets'][id] = net.speed or 1000

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
        values['jobs'] = [self.max_parallel_jobs, len(self.queue), len(self.jobs)]
        values['nets'] = {}
        values['processes'] = []

        for disk in psutil.disk_partitions():
            name = self.get_disk_name(disk[1])
            values['disks'][name] = psutil.disk_usage(disk[1]).used

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
            except psutil.Error:
                pass

        try:
            values['loadavg'] = os.getloadavg()
        except OSError:
            pass

        self.last_utilization = time.time()
        return values