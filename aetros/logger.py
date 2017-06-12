from __future__ import absolute_import
import sys
import time
import datetime
import six
from threading import Timer

class GeneralLogger(object):
    def __init__(self, logFD=None, job_backend=None, error=False):
        self.error = error
        self.job_backend = job_backend
        self.buffer = ''
        self.last_timer = None
        self.last_messages = ''

        self.terminal = sys.__stdout__ if error is False else sys.__stderr__
        self.logFD = logFD

    def get_time(self):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

        return st + '.' + str(ts % 1)[2:6]

    def get_line(self, line):
        if line:
            return "[%s] %s\n" % (self.get_time(), line)

        return line

    def fileno(self):
        return sys.__stdout__.fileno() if self.error is False else sys.__stderr__.fileno()

    def flush(self):
        if self.logFD:
            self.logFD.flush()

        self.terminal.flush()

    def send_to_buffer(self):
        self.last_timer = None

        if self.buffer:
            if self.logFD:
                self.logFD.write(self.buffer)

            if self.job_backend and self.job_backend.running:
                self.job_backend.write_log(self.buffer)

        self.buffer = ''

    def write(self, message):

        # if message == '\n':
        #     return

        message = six.text_type(message)

        # if not self.error:
        #     message = self.get_line(message)

        try:
            self.terminal.write(message)

            self.last_messages += message
            if len(self.last_messages) > 500 * 1024:
                self.last_messages = self.last_messages[-500 * 1024:]

            for char in message:
                if '\b' == char:
                    self.buffer = self.buffer[:-1]
                else:
                    self.buffer += char
        except:
            self.last_messages = ''
            self.buffer = ''
            pass

        if not self.last_timer:
            self.last_timer = Timer(1.0, self.send_to_buffer)
            self.last_timer.start()