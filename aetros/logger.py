from __future__ import absolute_import

import sys
import time
import datetime

import os
import six
from threading import Timer, Thread, Lock


class GeneralLogger(object):
    def __init__(self, job_backend, logger=None, error=False):
        self.error = error
        self.job_backend = job_backend
        self.buffer = ''
        self.last_timer = None
        self.last_messages = ''
        self.logger = logger
        self.lock = Lock()

        if not self.logger:
            self.logger = sys.__stdout__ if error is False else sys.__stderr__

    def fileno(self):
        return self.logger.fileno()

    def flush(self):
        self.logger.flush()
        self.send_buffer()

    def send_buffer(self):
        self.last_timer = None

        if self.buffer:
            buf = self.buffer
            self.buffer = ''

            if self.job_backend:
                self.job_backend.write_log(buf)

    def attach(self, buffer):
        """
        Read buffer until end (read() returns '') and sends it to self.logger and self.job_backend.

        :param buffer: a buffer instance with block read() method
        """
        def reader():
            while True:

                try:
                    # read() needs to block
                    # buf = os.read(buffer.fileno(), 4096)
                    buf = buffer.read(1)
                    if buf == six.b(''):
                        return

                    self.write(buf)
                except Exception as e:
                    #  we need to make sure, we continue to read otherwise the process of this buffer
                    # will block and we have a stuck process.
                    sys.__stderr__.write(str(e))
                    pass

        thread = Thread(target=reader)
        thread.daemon = True
        thread.start()

    def write(self, message):
        try:
            self.lock.acquire()

            if hasattr(message, 'decode'):
                # don't decode string again
                # necessary for Python3
                message = message.decode('utf-8')

            self.logger.write(message)

            self.last_messages += message
            if len(self.last_messages) > 20 * 1024:
                self.last_messages = self.last_messages[-20 * 1024:]

            for char in message:
                if '\b' == char or '\r' == char:
                    self.buffer = self.buffer[:-1]
                else:
                    self.buffer += char

            if not self.last_timer:
                self.last_timer = Timer(1.0, self.send_buffer)
                self.last_timer.start()
        except Exception as e:
            sys.__stderr__.write(str(e))
        finally:
            self.lock.release()