from __future__ import absolute_import

import sys
import six
from threading import Timer, Thread, Lock


def drain_stream(stream, decode='utf-8'):
    content = six.b('')

    while True:
        try:
            # read() needs to block
            # buf = os.read(buffer.fileno(), 4096)
            buf = stream.read(1)
            if buf == six.b(''):
                break
            content += buf
        except Exception:
            break

    if decode:
        return content.decode(decode)

    return content


class GeneralLogger(object):
    def __init__(self, redirect_to, job_backend=None):
        self.job_backend = job_backend
        self.buffer = ''
        self.last_timer = None
        self.last_messages = ''
        self.logger = redirect_to or sys.__stdout__
        self.lock = Lock()
        self.attach_last_messages = {}
        self.buffer_disabled = False

    def disable_buffer(self):
        self.buffer_disabled = True
        self.buffer = ''

    def clear_buffer(self):
        self.buffer = ''

    def fileno(self):
        return self.logger.fileno()

    def isatty(self):
        return self.logger.isatty()

    def flush(self):
        self.logger.flush()
        self.send_buffer()

    def send_buffer(self):
        self.last_timer = None

        if self.buffer:
            if self.job_backend:
                if self.job_backend.write_log(self.buffer):
                    self.buffer = ''

    def attach(self, buffer, read_line=False):
        """
        Read buffer until end (read() returns '') and sends it to self.logger and self.job_backend.

        :param buffer: a buffer instance with block read() method
        """

        bid = id(buffer)
        self.attach_last_messages[bid] = six.b('')

        lock = Lock()

        def reader():
            lock.acquire()
            while True:

                try:
                    # read() needs to block
                    # buf = os.read(buffer.fileno(), 4096)
                    if read_line:
                        buf = buffer.readline()
                    else:
                        buf = buffer.read(1)
                    if buf == six.b(''):
                        break

                    self.attach_last_messages[bid] += buf

                    if len(self.attach_last_messages[bid]) > 21 * 1024:
                        self.attach_last_messages[bid] = self.attach_last_messages[bid][-20 * 1024:]

                    self.write(buf)
                except ValueError as e:
                    if 'operation on closed' in e.message:
                        break

                except Exception as e:
                    # we need to make sure, we continue to read otherwise the process of this buffer
                    # will block and we have a stuck process.
                    sys.__stderr__.write(str(type(e))+': ' + str(e))
                    pass

            lock.release()

        thread = Thread(target=reader)
        thread.daemon = True
        thread.start()

        def wait():
            lock.acquire()
            self.send_buffer()
            lock.release()

        return wait

    def write(self, message):
        try:
            self.lock.acquire()

            if hasattr(message, 'decode'):
                # don't decode string again
                # necessary for Python3
                message = message.decode('utf-8')

            self.logger.write(message)
            self.logger.flush()

            self.last_messages += message
            if len(self.last_messages) > 20 * 1024:
                self.last_messages = self.last_messages[-20 * 1024:]

            if not self.buffer_disabled:
                for char in message:
                    if '\b' == char and self.buffer:
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
