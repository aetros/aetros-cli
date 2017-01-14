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

from aetros.backend import invalid_json_values, EventListener, parse_message


class Worker:
    """
    :type: event_listener : EventListener
    """

    def __init__(self, name, api_host, api_token, event_listener):
        self.name = name
        self.event_listener = event_listener
        self.api_token = api_token
        self.api_host = api_host
        self.active = True
        self.connected = True
        self.authenticated = False
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

                self.send_message({'login': self.api_token})
                messages = self.read_full_message(self.s)

                if "LOGGED_IN" in messages:
                    self.authenticated = True
                else:
                    print("Authentication with token %s against %s failed." % (self.api_token, self.api_host))
                    return

                self.send_message({'register_worker': self.name})
                messages = self.read_full_message(self.s)

                if "WORKER_NAME_USED" in messages:
                    print("Registration failed. Worker name already in use.")
                    return

                if "WORKER_REGISTERED" in messages:
                    self.registered = True
                else:
                    print("Registration failed.")
                    return

                break
            except socket.error as error:
                self.lock.release()
                print("Connection error during connecting to %s: %d: %s." % (self.api_host, error.errno, error.message))
                time.sleep(1)

        print("Connected to %s " % (self.api_host,))

    def thread(self):
        last_ping = 0

        while True:
            if self.connected and self.authenticated and self.registered:
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
        self.authenticated = False
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


class WorkerCommand:
    model = None
    job_model = None

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' worker')
        parser.add_argument('id', nargs='?', help='unique server name')
        parser.add_argument('--secure-key', help="Secure key. Alternatively use API_KEY environment varibale.")
        parser.add_argument('--server', help="Default aetros.com")

        parsed_args = parser.parse_args(args)

        if not parsed_args.id:
            parser.print_help()
            sys.exit()

        event_listener = EventListener()

        event_listener.on('job', self.job)

        worker = Worker(parsed_args.id, parsed_args.server, parsed_args.secure_key, event_listener)
        worker.start()

        while True:
            time.sleep(1)

    def job(self, message):
        pprint(message)
