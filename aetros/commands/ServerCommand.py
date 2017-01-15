from __future__ import absolute_import
from __future__ import print_function
import argparse
import json
import os
import socket
import sys
from pprint import pprint
from threading import Lock, Thread
import time

import select

import psutil

from aetros.backend import invalid_json_values, EventListener, parse_message


class Server:
    """
    :type: event_listener : EventListener
    """

    def __init__(self, api_host, server_api_token, event_listener):
        self.event_listener = event_listener
        self.server_api_token = server_api_token
        self.api_host = api_host
        self.active = True
        self.connected = True
        self.registered = False
        self.read_buffer = ''

        if not self.api_host:
            self.api_host = os.getenv('API_HOST')
            if not self.api_host or self.api_host == 'false':
                self.api_host = 'aetros.com'

        self.lock = Lock()

    def start(self):
        self.connect()

        self.thread = Thread(target=self.thread)
        self.thread.daemon = True
        self.thread.start()

    def connect(self):

        while self.active:
            # tries += 1
            # if tries > 3:
            #     print("Could not connect to %s. " % (self.api_host,))
            # break
            try:
                self.lock.acquire()
                self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.s.connect((self.api_host, 8051))
                self.connected = True
                self.lock.release()

                self.send_message({'register_server': self.server_api_token})
                messages = self.read_full_message(self.s)

                if "SERVER_ALREADY_REGISTERED" in messages:
                    print("Registration failed. This server is already registered.")
                    return

                if "SERVER_REGISTERED" in messages:
                    self.registered = True
                else:
                    print("Registration failed.")
                    return

                break
            except socket.error as error:
                self.lock.release()
                print("Connection error during connecting to %s: %d: %s." % (self.api_host, error.errno, error.message))
                time.sleep(1)

        self.event_listener.fire('registration')
        print("Connected to %s " % (self.api_host,))

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
                        # # send pending messages
                        # max_messages = 1
                        # for message in self.queue:
                        #     if not message['sending'] and not message['sent']:
                        #         max_messages += 1
                        #         self.send_message(message)
                        #         if max_messages > 10:
                        #             # not too much at once, so we have time to listen for incoming messages
                        #             break

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

                time.sleep(0.1)
            else:
                time.sleep(0.5)

    def close(self):
        self.active = False
        self.connected = False

        self.lock.acquire()
        self.s.shutdown(socket.SHUT_RDWR)
        self.s.close()
        self.lock.release()

    def handle_messages(self, messages):
        for message in messages:

            if 'job' in message:
                self.event_listener.fire('job', message['job'])

            if 'stop' in message:
                self.close()
                self.event_listener.fire('stop')

    def connection_error(self, error=None):
        if not self.active:
            return

        self.connected = False
        if error:
            print("Connection error: %d: %s" % (error.errno, error.message,))

        self.connected = False
        self.job_registered = False

        self.connect()

    def send_message(self, message):
        sent = False
        try:
            if isinstance(message, dict):
                message['sending'] = True

            msg = json.dumps(message, default=invalid_json_values)
            self.lock.acquire()
            self.s.sendall(msg + "\n")
            self.lock.release()

        except:
            self.lock.release()
            sent = False
            self.connected = False

        return sent

    def read_full_message(self, s):
        """
        Reads until we receive a message termination (\n)
        """
        message = ''

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

            message += chunk

            message, parsed = parse_message(message)
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
        self.jobs = []

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' worker')
        parser.add_argument('--secure-key', nargs='?',
                            help="Secure key of the server. Alternatively use API_KEY environment variable.")
        parser.add_argument('--server', help="Default aetros.com")

        parsed_args = parser.parse_args(args)

        if not parsed_args.secure_key:
            parser.print_help()
            sys.exit()

        event_listener = EventListener()

        event_listener.on('job', self.job)
        event_listener.on('registration', self.send_system_info)

        self.server = Server(parsed_args.server, parsed_args.secure_key, event_listener)
        self.server.start()

        while True:
            self.server.send_message({'type': 'utilization', 'values': self.collect_system_utilization()})
            time.sleep(0.5)

    def send_system_info(self, params):
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
        values['jobs'] = len(self.jobs)
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
                values['nets'][id]['upload'] = (net.bytes_sent - self.last_net[id]['sent']) / (time.time() - self.last_utilization)
                values['nets'][id]['download'] = (net.bytes_recv - self.last_net[id]['recv']) / (time.time() - self.last_utilization)

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

    def job(self, message):
        pprint(message)
