from __future__ import absolute_import

import sys
import traceback

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

    def attach(self, buffer, read_line=None):
        """
        Read buffer until end (read() returns '') and sends it to self.logger and self.job_backend.

        :param buffer: a buffer instance with block read() or readline() method
        :param read_line: callable or True to read line per line. If callable is given, it will be executed per line
                          and ignores does not redirect the line to stdout/logger when callable returns False.
        """

        bid = id(buffer)
        self.attach_last_messages[bid] = b''

        lock = Lock()

        def reader():
            lock.acquire()
            while True:

                try:
                    # read() needs to block
                    # buf = os.read(buffer.fileno(), 4096)
                    if read_line:
                        buf = buffer.readline()
                        if buf == b'':
                            break

                        if callable(read_line):
                            res = read_line(buf)
                            if res is False:
                                continue
                            elif res is not None:
                                buf = res
                                if hasattr(buf, 'encode'):
                                    buf = buf.encode("utf-8", 'replace')
                    else:
                        buf = buffer.read()
                        if buf == b'':
                            break

                    self.attach_last_messages[bid] += buf

                    if len(self.attach_last_messages[bid]) > 21 * 1024:
                        self.attach_last_messages[bid] = self.attach_last_messages[bid][-20 * 1024:]

                    self.write(buf)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    # we need to make sure, we continue to read otherwise the process of this buffer
                    # will block and we have a stuck process.
                    sys.__stderr__.write(traceback.format_exc() + '\n')
                    sys.__stderr__.flush()

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

            if b'' == message:
                return

            if hasattr(message, 'decode'):
                # don't decode string again
                # necessary for Python3
                message = message.decode("utf-8", 'replace')

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

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            sys.__stderr__.write(traceback.format_exc() + '\n')
            sys.__stderr__.flush()
        finally:
            if self.lock.locked():
                self.lock.release()
