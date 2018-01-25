from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import collections
import signal
import socket
import time

import msgpack
import requests
import simplejson
import sys

import six

from aetros.utils import invalid_json_values, prepend_signal_handler, create_ssh_stream
from threading import Thread, Lock
from aetros.const import __version__


class ApiClient:
    def __init__(self, api_host, api_key):
        self.host = api_host
        self.api_key = api_key

    def get(self, url, params=None, **kwargs):
        json_chunk = kwargs.get('json')
        if json_chunk and not isinstance(json_chunk, str):
            kwargs['json'] = simplejson.loads(simplejson.dumps(json_chunk, default=invalid_json_values),
                                              object_pairs_hook=collections.OrderedDict)

        return requests.get(self.get_url(url), params=params, **kwargs)

    def post(self, url, data=None, **kwargs):
        json_chunk = kwargs.get('json')
        if json_chunk and not isinstance(json_chunk, str):
            kwargs['json'] = simplejson.loads(simplejson.dumps(json_chunk, default=invalid_json_values),
                                              object_pairs_hook=collections.OrderedDict)

        return requests.post(self.get_url(url), data=data, **kwargs)

    def put(self, url, data=None, **kwargs):
        json_chunk = kwargs.get('json')
        if json_chunk and not isinstance(json_chunk, str):
            kwargs['json'] = simplejson.loads(simplejson.dumps(json_chunk, default=invalid_json_values),
                                              object_pairs_hook=collections.OrderedDict)

        return requests.put(self.get_url(url), data=data, **kwargs)

    def get_url(self, affix):

        url = 'http://%s/api/%s' % (self.host, affix)

        if self.api_key:
            if '?' in url:
                url += '&token=' + self.api_key
            else:
                url += '?token=' + self.api_key

        return url


class BackendClient:
    def __init__(self, config, event_listener, logger):
        self.config = config
        self.host = config['host']
        self.go_offline_on_first_failed_attempt = True

        self.event_listener = event_listener
        self.logger = logger
        self.message_id = 0
        self.sync_status = {}

        self.api_key = None
        self.job_id = None

        self.queues = {}
        self.ssh_stream = {}
        self.ssh_channel = {}
        self.thread_read_instances = {}
        self.thread_write_instances = {}
        self.stop_on_empty_queue = {}

        self.lock = Lock()
        self.connection_errors = 0
        self.connection_tries = 0
        self.in_connecting = False
        self.write_speeds = {}
        self.read_speeds = []

        # indicates whether we are offline or not, means not connected to the internet and
        # should not establish a connection to Aetros.
        self.online = True

        # Whether the client is active and should do things.
        self.active = False
        self.expect_close = False
        self.external_stopped = False

        # the connection is authenticated against the server and ready to send stuff
        self.registered = {}

        # the actual connection is established
        self.connected = {}

        self.was_connected_once = {}
        self.read_unpacker = msgpack.Unpacker(encoding='utf-8')

    def on_sigint(self, sig, frame):
        # when connections breaks, we do not reconnect
        self.expect_close = True

    def start(self, channels=None):
        self.active = True
        prepend_signal_handler(signal.SIGINT, self.on_sigint)

        self.queues = {}
        self.thread_read_instances = {}
        self.thread_write_instances = {}
        self.stop_on_empty_queue = {}
        self.ssh_stream = {}
        self.was_connected_once = {}

        if not channels:
            channels = ['']

        for channel in channels:
            self.queues[channel] = []

            self.ssh_stream[channel] = None
            self.ssh_channel[channel] = None
            self.was_connected_once[channel] = False
            self.stop_on_empty_queue[channel] = False
            self.write_speeds[channel] = []

            self.thread_read_instances[channel] = Thread(target=self.thread_read, args=[channel])
            self.thread_read_instances[channel].daemon = True
            self.thread_read_instances[channel].start()

            self.thread_write_instances[channel] = Thread(target=self.thread_write, args=[channel])
            self.thread_write_instances[channel].daemon = True
            self.thread_write_instances[channel].start()

    def on_connect(self, reconnect, channel):
        pass

    def go_offline(self):
        if not self.online:
            return

        self.event_listener.fire('offline')
        self.online = False

    def connect(self, channel):
        """
        In the write-thread we detect that no connection is living anymore and try always again.
        Up to the 3 connection try, we report to user. We keep trying but in silence.
        Also, when more than 10 connection tries are detected, we delay extra 15 seconds.
        """
        if self.connection_tries > 10:
            time.sleep(15)

        if self.in_connecting:
            return False

        self.in_connecting = True

        self.logger.debug('[%s] Wanna connect ...' % (channel, ))

        try:
            if self.is_connected(channel) or not self.online:
                return True

            self.lock.acquire()

            self.connected[channel] = False
            self.registered[channel] = False
            self.ssh_stream[channel] = False
            self.ssh_channel[channel] = False
            self.write_speeds[channel] = []
            messages = None
            stderrdata = ''

            try:
                if not self.ssh_stream[channel]:
                    self.logger.debug('[%s] Open ssh connection' % (channel, ))
                    self.ssh_stream[channel] = create_ssh_stream(self.config, exit_on_failure=False)

                self.logger.debug('[%s] open channel' % (channel, ))

                self.ssh_channel[channel] = self.ssh_stream[channel].get_transport().open_session()
                self.ssh_channel[channel].exec_command('stream')

                # self.ssh_stream_stdins[channel],\
                #     self.ssh_stream_stdouts[channel],\
                #     stderr = self.ssh_stream[channel].exec_command('stream')
            except Exception:
                raise
            finally:
                self.lock.release()

            if self.ssh_channel[channel]:
                messages = self.wait_for_at_least_one_message(channel)

            if not messages:
                stderrdata = self.ssh_channel[channel].recv_stderr().decode("utf-8").strip()
            else:
                self.logger.debug('[%s] opened and received %d messages' % (channel, len(messages)))


                self.connected[channel] = True
                self.registered[channel] = self.on_connect(self.was_connected_once[channel], channel)

            if not self.registered[channel]:
                self.logger.debug("[%s] Client: registration failed. stderrdata: %s" % (channel, stderrdata))
                self.connected[channel] = False

                try:
                    self.logger.debug('[%s] Client: ssh_tream close' % (channel, ))
                    self.ssh_stream[channel].close()
                except (KeyboardInterrupt, SystemExit):
                    raise

                self.connection_tries += 1
                if not self.was_connected_once[channel] and self.go_offline_on_first_failed_attempt:
                    # initial try needs to be online, otherwise we go offline
                    self.go_offline()

                if stderrdata:
                    if 'Connection refused' not in stderrdata and 'Permission denied' not in stderrdata:
                        self.logger.error(stderrdata)

                if 'Permission denied' in stderrdata:
                    if self.connection_tries < 3:
                        self.logger.warning("Access denied. Did you setup your SSH public key correctly "
                                            "and saved it in your AETROS Trainer user account?")

                    self.close()
                    sys.exit(1)

                self.connection_error(channel, "Connection error during connecting to %s: %s" % (self.host, str(stderrdata)))
            else:
                self.was_connected_once[channel] = True

        except Exception as error:
            self.connection_error(channel, error)
        finally:
            self.in_connecting = False

        return self.is_connected(channel)

    # def debug(self):
    #     sent = len(filter(lambda x: x['_sent'], self.queue))
    #     sending = len(filter(lambda x: x['_sending'], self.queue))
    #     open = len(filter(lambda x: not x['_sending'], self.queue))
    #     self.logger.debug("%d sent, %d in sending, %d open " % (sent, sending, open))

    def end(self):
        self.expect_close = True

        for channel in six.iterkeys(self.ssh_channel):
            self.send_message({'type': 'end'}, channel)

        self.wait_for_close()

    def connection_error(self, channel, error=None):
        if not self.active:
            # we don't care when we're not active
            return

        # give it some free time
        time.sleep(0.1)

        # make sure ssh connection is closed, so we can recover
        try:
            if self.ssh_stream[channel]:
                self.logger.debug('[%s] Client: ssh_tream close' % (channel,))
                self.ssh_stream[channel].close()
        except (KeyboardInterrupt, SystemExit):
            raise

        if self.expect_close:
            # we expected the close, so ignore the error
            return

        # needs to be set before logger.error, since they can call send_message again
        self.connected = {}
        self.registered = {}

        if socket is None:
            # python interpreter is already dying, so quit
            return

        message = "Connection error"

        if error:
            import traceback
            self.logger.debug(traceback.format_exc())

            if hasattr(error, 'message'):
                self.logger.error(message + ": " + str(error.message))
            else:
                self.logger.error(message + ": " + str(error))

            if 'No authentication methods available' in str(error):
                self.logger.error("Make sure you have authenticated your machine correctly using "
                                  "'aetros authenticate'.")
        else:
            self.logger.error(message)

        self.event_listener.fire('disconnect')
        self.connection_errors += 1

    def thread_write(self, channel):
        while self.active:
            if self.online:
                if self.is_connected(channel) and self.is_registered(channel) and self.queues[channel]:
                    message = self.queues[channel][0]

                    try:
                        sent = []

                        if message['_sending'] and not message['_sent']:
                            message['_sending'] = False

                        if not self.is_connected(channel) or not self.is_registered(channel):
                            # additional check to make sure there's no race condition
                            self.logger.debug('[%s] break while sending' % (channel,))
                            break

                        if not message['_sending'] and not message['_sent']:
                            self.send_message(message, channel)
                            sent.append(message)

                        self.lock.acquire()
                        if message in self.queues[channel]:
                            self.queues[channel].remove(message)
                        self.lock.release()

                    except Exception as e:
                        self.logger.debug('[%s] Closed write thread: exception. %d messages left'
                                          % (channel, len(self.queues[channel]), ))
                        self.connection_error(channel, e)

                if self.stop_on_empty_queue[channel]:
                    if len(self.queues[channel]) == 0 or not self.is_connected(channel) or \
                            not self.is_registered(channel):
                        self.logger.debug('[%s] Closed write thread: ended. %d messages left'
                                          % (channel, len(self.queues[channel]),))
                        return

                if self.active and not self.is_connected(channel) and not self.expect_close:
                    if not self.connect(channel):
                        time.sleep(5)

            time.sleep(0.1)

        self.logger.debug('[%s] Closed write thread: disconnect. %d messages left' % (channel, len(self.queues[channel]), ))

    def thread_read(self, channel):
        while self.active:
            if self.online:
                if self.is_connected(channel) and self.is_registered(channel):
                    try:
                        # this blocks
                        messages = self.read(channel)

                        if messages is not None:
                            self.logger.debug("[%s] Client: handle message: %s" % (channel, str(messages)))
                            self.handle_messages(messages)

                        if self.stop_on_empty_queue[channel]:
                            return
                    except Exception as e:
                        self.logger.debug('[%s] Closed read thread: exception' % (channel, ))
                        self.connection_error(channel, e)

            time.sleep(0.01)

        self.logger.debug('[%s] Closed read thread: ended' % (channel, ))

    def _end_channel(self, channel):
        """
        Soft end of ssh channel. End the writing thread as soon as the message queue is empty.
        """
        self.stop_on_empty_queue[channel] = True

        # by joining the we wait until its loop finishes.
        # it won't loop forever since we've set self.stop_on_empty_queue=True
        write_thread = self.thread_write_instances[channel]

        try:
            while True:
                if write_thread.isAlive():
                    write_thread.join(0.5)

                if not write_thread.isAlive():
                    break

                time.sleep(0.5)

        except (KeyboardInterrupt, SystemExit):
            raise

    def wait_sending_last_messages(self):
        if self.active and self.online:
            self.logger.debug("client sends last %s messages ..."
                              % ([str(i) + ':' + str(len(x)) for i, x in six.iteritems(self.queues)],))

            # send all missing messages

            # by joining we wait until its loop finish.
            # it won't loop forever since we've set self.stop_on_empty_queue=True
            for channel in six.iterkeys(self.ssh_channel):
                if channel != '':
                    self._end_channel(channel)

            # last is control channel
            self._end_channel('')

    def wait_for_close(self):
        if not (self.active and self.online):
            return

        self.active = False

        i = 0
        try:
            for channel, file in six.iteritems(self.ssh_channel):
                while file and not file.closed:
                    i += 1
                    time.sleep(0.1)
                    if i % 50 == 0:
                        self.logger.warning("[%s] We are still waiting for connection closing on server side."
                                            % (channel, ))
        except (SystemExit, KeyboardInterrupt):
            raise

        self.online = False

    def close(self):
        self.active = False
        self.connected = {}
        self.registered = {}

        for channel, stream in six.iteritems(self.ssh_stream):
            try:
                if stream:
                    self.logger.debug('[%s] Client: ssh_tream close' % (channel, ))
                    stream.close()
            except (KeyboardInterrupt, SystemExit):
                raise

        if self.online:
            self.event_listener.fire('close')

        self.ssh_stream = {}
        self.online = False

    def is_connected(self, channel):
        return channel in self.connected and self.connected[channel]

    def is_registered(self, channel):
        return channel in self.registered and self.registered[channel]

    def send(self, data, channel='', important=False):
        if not (self.active and self.online):
            # It's important to queue anything when active and online
            # as we would lose information in git streams.
            return

        if self.stop_on_empty_queue[channel]:
            # make sure, we don't add new one
            return

        self.message_id += 1

        message = {}
        message['_id'] = self.message_id

        if 'type' in data:
            message['type'] = data['type']

        if 'path' in data:
            message['path'] = data['path']

        message['_data'] = msgpack.packb(data, default=invalid_json_values)
        message['_total'] = len(message['_data'])
        message['_bytes_sent'] = 0
        message['_sending'] = False
        message['_sent'] = False

        if important:
            self.queues[channel].insert(0, message)
        else:
            self.queues[channel].append(message)

    def send_message(self, message, channel):
        """
        Internal. Sends the actual message from a queue entry.
        """
        if not self.is_connected(channel):
            return False

        message['_sending'] = True

        if '_data' in message:
            data = message['_data']
            total = message['_total']
        else:
            data = msgpack.packb(message, default=invalid_json_values)
            total = len(data)
            message['_bytes_sent'] = 0
            message['_id'] = -1

        sent_sizes = []

        def report():
            if not self.write_speeds[channel] or not sent_sizes:
                return
            # speed = sum(self.write_speeds) / len(self.write_speeds)
            # average_size = sum(sent_sizes) / len(sent_sizes)
            # print("[%s] Speed is at %.3f kb/s (average send() size=%.3fkb), %.3fkb of %.3fkb sent, id=%s" % (
            #        channel, speed / 1024, average_size / 1024, message['_bytes_sent'] / 1024,
            #        total / 1024, message['_id']))

        try:
            while data:
                start = time.time()
                # buf = data[:BUFFER_SIZE]
                bytes_sent = self.ssh_channel[channel].send(data)

                sent_sizes.append(bytes_sent)
                data = data[bytes_sent:]
                message['_bytes_sent'] += bytes_sent

                # data = data[BUFFER_SIZE:]
                # bytes_sent += len(buf)
                end = time.time()
                self.write_speeds[channel].append( bytes_sent / (end-start) )

                if len(self.write_speeds[channel]) > 20:
                    self.write_speeds[channel] = self.write_speeds[channel][10:]
                    sent_sizes = sent_sizes[10:]

            report()
            message['_sent'] = True
            return True

        except (KeyboardInterrupt, SystemExit):
            if message['_sent']:
                return total

            return False

        except Exception as error:
            self.connection_error(channel, error)
            return False

        finally:
            self.write_speeds[channel] = []

    def handle_messages(self, messages):
        self.lock.acquire()
        try:
            for message in messages:
                if not isinstance(message, dict):
                    continue

                if 'a' in message:
                    if not self.external_stopped and 'stop' == message['a']:
                        self.external_stopped = True
                        self.event_listener.fire('stop', message['force'])
        finally:
            self.lock.release()

    def wait_for_at_least_one_message(self, channel):
        """
        Reads until we receive at least one message we can unpack. Return all found messages.
        """

        unpacker = msgpack.Unpacker(encoding='utf-8')

        while True:
            try:
                start = time.time()
                chunk = self.ssh_channel[channel].recv(1024)
                end = time.time()
                self.read_speeds.append( len(chunk) / (end-start) )
                if len(self.read_speeds) > 20:
                    self.read_speeds = self.read_speeds[10:]

                if chunk == '':
                    # happens only when connection broke. If nothing is to be received, it hangs instead.
                    self.connection_error(channel, 'Connection broken w')
                    return False
            except Exception as error:
                self.connection_error(channel, error)
                raise

            unpacker.feed(chunk)

            messages = [m for m in unpacker]
            if messages:
                return messages

    def read(self, channel):
        """
        Reads from the socket and tries to unpack the message. If successful (because msgpack was able to unpack)
        then we return that message. Else None. Keep calling .read() when new data is available so we try it
        again.
        """

        if not self.ssh_channel[channel].recv_ready():
            return

        try:
            start = time.time()
            chunk = self.ssh_channel[channel].recv(1024)
            end = time.time()

            self.read_speeds.append(len(chunk) / (end-start))
            if len(self.read_speeds) > 20:
                self.read_speeds = self.read_speeds[10:]

        except Exception as error:
            self.connection_error(channel, error)
            raise

        if chunk == '':
            # socket connection broken
            self.connection_error(channel, 'Connection broken')
            return None

        # self.read_buffer.seek(0, 2) #make sure we write at the end
        self.read_unpacker.feed(chunk)

        # self.read_buffer.seek(0)
        messages = [m for m in self.read_unpacker]

        return messages if messages else None


class JobClient(BackendClient):
    def __init__(self, config, event_listener, logger):
        BackendClient.__init__(self, config, event_listener, logger)
        self.model_name = None
        self.name = None

    def configure(self, model_name, job_id, name):
        self.model_name = model_name
        self.job_id = job_id
        self.name = name

    def on_connect(self, reconnect, channel):
        self.send_message({
            'type': 'register_job_worker',
            'model': self.model_name,
            'job': self.job_id,
            'reconnect': reconnect,
            'version': __version__,
            'name': self.name + channel
        }, channel)

        self.logger.debug("[%s] Wait for job client registration for %s" % (channel, self.name))
        messages = self.wait_for_at_least_one_message(channel)
        self.logger.debug("[%s] Got %s" % (channel, str(messages)))

        if not messages:
            self.event_listener.fire('registration_failed', {'reason': 'No answer received.'})
            return False

        message = messages.pop(0)
        self.logger.debug("[%s] Client: handle message: %s" % (channel, str(message)))
        if isinstance(message, dict) and 'a' in message:

            if 'aborted' == message['a']:
                if channel == '':
                    self.logger.error("[%s] Job aborted or deleted meanwhile. Exiting" % (channel,))
                    self.event_listener.fire('aborted')
                self.active = False
                return False

            if 'registration_failed' == message['a']:
                if channel == '':
                    self.event_listener.fire('registration_failed', {'reason': message['reason']})
                return False

            if 'registered' == message['a']:
                self.registered[channel] = True
                if channel == '':
                    self.event_listener.fire('registration')
                self.handle_messages(messages)
                return True

        self.logger.error("[%s] Registration of job %s failed." % (channel, self.job_id,))
        return False

    def handle_messages(self, messages):

        BackendClient.handle_messages(self, messages)
        for message in messages:
            if self.external_stopped:
                continue

            if not isinstance(message, dict):
                continue

            if 'a' in message and 'parameter-changed' == message['a']:
                self.event_listener.fire('parameter_changed', {'values': message['values']})

            if 'type' in message and 'action' == message['type']:
                self.event_listener.fire('action', message)