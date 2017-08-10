from __future__ import absolute_import

import sys
import time
import datetime
import six
from threading import Timer, Thread


class GeneralLogger(object):
    def __init__(self, job_backend, logger=None, error=False):
        self.error = error
        self.job_backend = job_backend
        self.buffer = ''
        self.last_timer = None
        self.last_messages = ''
        self.logger = logger or (sys.__stdout__ if error is False else sys.__stderr__)

    def get_time(self):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

        return st + '.' + str(ts % 1)[2:6]

    def get_line(self, line):
        if line:
            return "[%s] %s\n" % (self.get_time(), line)

        return line

    def fileno(self):
        return self.logger.fileno()

    def flush(self):
        self.logger.flush()
        self.send_buffer()

    def send_buffer(self):
        self.last_timer = None

        if self.buffer:
            if self.job_backend:
                self.job_backend.write_log(self.buffer)

        self.buffer = ''

    def attach(self, buffer):
        """
        Read buffer until end (read() returns '') and sends it to self.logger and self.job_backend.

        :param buffer: a buffer instance with block read() method
        """
        def reader():
            while True:

                # read() needs to block
                buf = buffer.read(1)
                if buf == '':
                    return

                self.write(buf.decode("utf-8"))

        thread = Thread(target=reader)
        thread.daemon = True
        thread.start()

    def write(self, message):
        try:
            message = six.text_type(message)
        except:
            pass

        try:
            self.logger.write(message)

            self.last_messages += message
            if len(self.last_messages) > 20 * 1024:
                self.last_messages = self.last_messages[-20 * 1024:]

            for char in message:
                if '\b' == char or '\r' == char:
                    self.buffer = self.buffer[:-1]
                else:
                    self.buffer += char
        except:
            self.last_messages = ''
            self.buffer = ''
            pass

        if not self.last_timer:
            self.last_timer = Timer(1.0, self.send_buffer)
            self.last_timer.start()