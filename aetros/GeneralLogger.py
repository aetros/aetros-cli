import sys
import time
import datetime

class GeneralLogger():
    def __init__(self, job, logFD=None, aetros_backend=None, error=False):
        self.error = error
        self.job = job
        self.aetros_backend = aetros_backend

        self.terminal = sys.stdout if error is False else sys.stderr
        self.logFD = logFD

    def get_time(self):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

        return st + '.' + str(ts%1)[2:6]

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

        message = unicode(message)

        # if not self.error:
        #     message = self.get_line(message)

        self.terminal.write(message)
        if self.logFD:
            self.logFD.write(message)

        if self.aetros_backend:
            self.aetros_backend.write_log(self.job['id'], message)
