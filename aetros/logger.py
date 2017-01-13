from __future__ import absolute_import
import sys
import time
import datetime
import six


class GeneralLogger():
    def __init__(self, logFD=None, job_backend=None, error=False):
        self.error = error
        self.job_backend = job_backend

        self.terminal = sys.stdout if error is False else sys.stderr
        self.logFD = logFD

    def get_time(self):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

        return st + '.' + str(ts % 1)[2:6]

    def get_line(self, line):
        if line:
            return "[%s] %s\n" % (self.get_time(), line)

        return line

    def flush(self):
        if self.logFD:
            self.logFD.flush()

        self.terminal.flush()

    def write(self, message):

        # if message == '\n':
        #     return

        message = six.text_type(message)

        # if not self.error:
        #     message = self.get_line(message)

        self.terminal.write(message)
        if self.logFD:
            self.logFD.write(message)

        if self.job_backend:
            self.job_backend.write_log(message)
