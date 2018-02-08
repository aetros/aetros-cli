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
from paramiko.ssh_exception import NoValidConnectionsError

from aetros.utils import invalid_json_values, prepend_signal_handler, create_ssh_stream, is_debug, is_debug2
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
        self.channel_closed = {}

        self.bytes_sent = 0
        self.bytes_total = 0
        self.bytes_speed = 0

        self.lock = Lock()
        self.channel_lock = {}
        self.queue_lock = {}
        self.connection_errors = 0
        self.connection_tries = 0
        self.in_connecting = {}
        self.write_speeds = []
        self.read_speeds = []

        # indicates whether we are offline or not, means not connected to the internet and
        # should not establish a connection to Aetros.
        self.online = None

        # Whether the client is active and should do things.
        self.active = False
        self.expect_close = False
        self.external_stopped = False

        # the connection is authenticated against the server and ready to send stuff
        self.registered = {}

        # the actual connection is established
        self.connected = {}

        self.was_connected_once = {}
        self.connected_since = {}
        self.read_unpacker = msgpack.Unpacker(encoding='utf-8')

    def on_sigint(self, sig, frame):
        # when connections breaks, we do not reconnect
        self.expect_close = True

    def start(self, channels=None):
        if self.active:
            return

        self.logger.debug('Client start')
        self.active = True
        prepend_signal_handler(signal.SIGINT, self.on_sigint)

        self.queues = {}
        self.thread_read_instances = {}
        self.thread_write_instances = {}
        self.stop_on_empty_queue = {}
        self.connected = {}
        self.registered = {}
        self.ssh_stream = {}
        self.was_connected_once = {}

        if not channels:
            channels = ['']

        for channel in channels:
            self.queues[channel] = []

            self.ssh_stream[channel] = None
            self.ssh_channel[channel] = None
            self.connected[channel] = None
            self.registered[channel] = None
            self.connected_since[channel] = 0
            self.was_connected_once[channel] = False
            self.stop_on_empty_queue[channel] = False
            self.channel_lock[channel] = Lock()
            self.queue_lock[channel] = Lock()
            self.in_connecting[channel] = False
            self.channel_closed[channel] = False

            self.thread_read_instances[channel] = Thread(target=self.thread_read, args=[channel])
            self.thread_read_instances[channel].daemon = True
            self.thread_read_instances[channel].start()

            self.thread_write_instances[channel] = Thread(target=self.thread_write, args=[channel])
            self.thread_write_instances[channel].daemon = True
            self.thread_write_instances[channel].start()

        while True:
            # check if all was_connected_once is not-None (True or False)
            all_set = all(x is not None for x in six.itervalues(self.registered))

            if all_set:
                # When all True then success, if not then unsuccessful
                self.online = all(six.itervalues(self.registered))

                return self.online

            time.sleep(0.1)

    def on_connect(self, reconnect, channel):
        pass

    def go_offline(self):
        if self.online is False:
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
            time.sleep(10)

        if self.in_connecting[channel]:
            return False

        self.in_connecting[channel] = True

        self.logger.debug('[%s] Wanna connect ...' % (channel, ))

        try:
            if self.is_connected(channel) or self.online is False:
                if self.is_connected(channel):
                    self.logger.debug('[%s] Already connected' % (channel, ))
                if self.online is False:
                    self.logger.debug('[%s] self.online=False' % (channel, ))

                return True

            self.channel_lock[channel].acquire()

            self.connected[channel] = None
            self.registered[channel] = None
            self.ssh_stream[channel] = False
            self.ssh_channel[channel] = False
            messages = None
            stderrdata = ''

            try:
                if not self.ssh_stream[channel]:
                    self.logger.debug('[%s] Open ssh connection' % (channel, ))
                    self.ssh_stream[channel] = create_ssh_stream(self.config, exit_on_failure=False)

                self.logger.debug('[%s] open channel' % (channel, ))

                self.ssh_channel[channel] = self.ssh_stream[channel].get_transport().open_session()
                self.ssh_channel[channel].exec_command('stream')
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                self.logger.debug('[%s] connection failed: %s'  % (channel, str(e)))
                return False
            finally:
                self.channel_lock[channel].release()

            if self.ssh_channel[channel]:
                messages = self.wait_for_at_least_one_message(channel)

            if not messages:
                stderrdata = self.ssh_channel[channel].recv_stderr().decode("utf-8").strip()
                self.connected[channel] = False
            else:
                self.logger.debug('[%s] opened and received %d messages' % (channel, len(messages)))
                self.connected[channel] = True
                self.registered[channel] = self.on_connect(self.was_connected_once[channel], channel)
                self.connected_since[channel] = time.time()

                if channel == '' and self.registered[channel] and self.was_connected_once[channel]:
                    self.logger.info("Successfully reconnected.")

            if not self.registered[channel]:
                # make sure to close channel and connection first
                try:
                    self.ssh_channel[channel] and self.ssh_channel[channel].close()
                except: pass

                try:
                    self.ssh_stream[channel] and self.ssh_stream[channel].close()
                except: pass

                self.logger.debug("[%s] Client: registration failed. stderrdata: %s" % (channel, stderrdata))
                self.connected[channel] = False

                try:
                    self.logger.debug('[%s] Client: ssh_tream close due to registration failure' % (channel, ))
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
            self.in_connecting[channel] = False

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
            if self.ssh_channel[channel]:
                self.ssh_channel[channel].close()
        except (KeyboardInterrupt, SystemExit):
            raise

        try:
            if self.ssh_stream[channel]:
                self.logger.debug('[%s] Client: ssh_stream close due to connection error' % (channel,))
                self.ssh_stream[channel].close()
        except (KeyboardInterrupt, SystemExit):
            raise

        if self.expect_close:
            # we expected the close, so ignore the error
            return

        # needs to be set before logger.error, since they can call send_message again
        self.connected[channel] = False
        self.registered[channel] = False

        if socket is None:
            # python interpreter is already dying, so quit
            return

        if channel != '':
            # we don't care about the file channel,
            # it will reconnect anyway
            return

        since = 0
        if self.connected_since[channel]:
            since = time.time() - self.connected_since[channel]

        message = "[%s] Connection error (connected for %d seconds) " % (channel, since)

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
            if self.online is not False:
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

                        self.queue_lock[channel].acquire()
                        if message in self.queues[channel]:
                            if message['_sent']:
                                self.queues[channel].remove(message)
                        self.queue_lock[channel].release()

                    except Exception as e:
                        self.logger.debug('[%s] Closed write thread: exception. %d messages left'
                                          % (channel, len(self.queues[channel]), ))
                        self.connection_error(channel, e)
                else:
                    time.sleep(0.1)

                if self.stop_on_empty_queue[channel]:
                    if len(self.queues[channel]) == 0 or not self.is_connected(channel) or \
                            not self.is_registered(channel):
                        self.logger.debug('[%s] Closed write thread: ended. %d messages left'
                                          % (channel, len(self.queues[channel]),))
                        return

                if self.active and not self.is_connected(channel) and not self.expect_close:
                    if not self.connect(channel):
                        time.sleep(1)

        self.logger.debug('[%s] Closed write thread: disconnect. %d messages left' % (channel, len(self.queues[channel]), ))

    def thread_read(self, channel):
        while self.active:
            if self.online is not False:
                if self.is_connected(channel) and self.is_registered(channel):
                    try:
                        # this blocks if we have data
                        messages = self.read(channel)

                        if messages is not None:
                            self.logger.debug("[%s] Client: handle message: %s" % (channel, str(messages)))
                            self.handle_messages(channel, messages)

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
        """
        Requests all channels to close and waits for it.
        """
        if self.active and self.online is not False:
            self.logger.debug("client sends last %s messages ..."
                              % ([str(i) + ':' + str(len(x)) for i, x in six.iteritems(self.queues)],))

            for channel, messages in six.iteritems(self.queues):
                for idx, message in enumerate(messages):
                    self.logger.debug("[%s] %d: %s" % (channel, idx, str(message)[0:120]))

            # send all missing messages

            # by joining we wait until its loop finish.
            # it won't loop forever since we've set self.stop_on_empty_queue=True
            for channel in six.iterkeys(self.ssh_channel):
                if channel != '':
                    self._end_channel(channel)

            # last is control channel
            self._end_channel('')

    def is_channel_open(self, channel):
        return not self.channel_closed[channel]

    def wait_until_queue_empty(self, channels, report=True, clear_end=True):
        """
        Waits until all queues of channels are empty.
        """
        state = {'message': ''}

        self.logger.debug("wait_until_queue_empty: report=%s %s"
                          % (str(report), str([channel+':'+str(len(self.queues[channel])) for channel in channels]), ))
        queues = []
        for channel in channels:
            queues += self.queues[channel][:]

        def print_progress():
            if report:
                self.logger.debug("all_empty=%s" % (str(all_empty),))

                sys.__stderr__.write('\b' * len(state['message']))
                sys.__stderr__.write("\033[K")

                state['message'] = "%.2f kB/s // %.2fkB of %.2fkB // %.2f%%" \
                          % (self.bytes_speed / 1024, self.bytes_sent / 1024, self.bytes_total / 1024,
                            (self.bytes_sent / self.bytes_total * 100) if self.bytes_total else 0)

                sys.__stderr__.write(state['message'])
                sys.__stderr__.flush()

        while True:
            all_empty = all(m['_sent'] for m in queues)

            print_progress()

            if all_empty:
                break

            time.sleep(0.2)

        print_progress()

        if report and clear_end:
            sys.__stderr__.write('\b' * len(state['message']))
            sys.__stderr__.write("\033[K")
            sys.__stderr__.flush()

    def wait_for_close(self):
        if not self.active or self.online is False:
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
                    self.logger.debug('[%s] Client: ssh_tream close due to close call' % (channel, ))
                    stream.close()
            except (KeyboardInterrupt, SystemExit):
                raise

        if self.online is True:
            self.event_listener.fire('close')

        self.ssh_stream = {}
        self.online = False

    def is_connected(self, channel):
        return channel in self.connected and self.connected[channel]

    def is_registered(self, channel):
        return channel in self.registered and self.registered[channel]

    def send(self, data, channel='', important=False):
        if not self.active or self.online is False:
            # It's important to queue anything when active and online
            # as we would lose information in git streams.
            return

        self.queue_lock[channel].acquire()

        try:
            if self.channel_closed[channel]:
                # make sure, we don't add new one
                self.logger.debug('Warning: channel %s got message although closed: %s' % (channel, str(data)[:150]))
                return

            if self.stop_on_empty_queue[channel]:
                # make sure, we don't add new one
                self.logger.debug('Warning: channel %s got message although requested to stop: %s'
                                  % (channel, str(data)[:150]))
                return

            self.message_id += 1

            if is_debug2():
                sys.__stderr__.write("JobBackend:send(%s, %s, %s)\n" % (str(data)[0:180], str(channel), str(important)))
                sys.__stderr__.flush()

            message = {}
            message['_id'] = self.message_id

            if 'type' in data:
                message['type'] = data['type']

            if 'path' in data:
                message['path'] = data['path']

            if 'type' in data and data['type'] == 'git-unpack-objects':
                # extract to send it to the UI, to display what is currently being uploaded
                message['objects'] = data['objects']
                del data['objects']

            message['_data'] = msgpack.packb(data, default=invalid_json_values)
            message['_total'] = len(message['_data'])
            message['_bytes_sent'] = 0
            message['_sending'] = False
            message['_sent'] = False

            self.bytes_total += message['_total']

            if important:
                self.queues[channel].insert(0, message)
            else:
                self.queues[channel].append(message)

            return message['_total']
        finally:
            self.queue_lock[channel].release()

    def send_message(self, message, channel):
        """
        Internal. Sends the actual message from a queue entry.
        """
        if not self.is_connected(channel):
            return False

        message['_sending'] = True

        if '_data' in message:
            data = message['_data']
        else:
            data = msgpack.packb(message, default=invalid_json_values)
            self.bytes_total += len(data)
            message['_bytes_sent'] = 0
            message['_id'] = -1

        if is_debug2():
            sys.__stderr__.write("[%s] send message: %s\n" % (channel, str(msgpack.unpackb(data))[0:180]))

        try:
            while data:
                start = time.time()
                bytes_sent = self.ssh_channel[channel].send(data)

                data = data[bytes_sent:]
                message['_bytes_sent'] += bytes_sent
                self.bytes_sent += bytes_sent

                end = time.time()
                self.write_speeds.append(bytes_sent / (end-start))

                speeds_len = len(self.write_speeds)
                if speeds_len:
                    self.bytes_speed = sum(self.write_speeds) / speeds_len
                    if speeds_len > 10:
                        self.write_speeds = self.write_speeds[5:]

            message['_sent'] = True
            return True

        except (KeyboardInterrupt, SystemExit):
            if message['_sent']:
                return message['_bytes_sent']

            return False

        except Exception as error:
            self.connection_error(channel, error)
            return False

    def handle_messages(self, channel, messages):
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

                if chunk == b'':
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

        if chunk == b'':
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
        self.logger.debug("[%s] Got onconnect %s" % (channel, str(messages)))

        if not messages:
            self.event_listener.fire('registration_failed', {'reason': 'No answer received.'})
            return False

        message = messages.pop(0)
        self.logger.debug("[%s] Client: onconnect handle message: %s" % (channel, str(message)))
        if isinstance(message, dict) and 'a' in message:

            if 'aborted' == message['a']:
                self.logger.error("[%s] Job aborted or deleted meanwhile. Exiting" % (channel,))
                if channel == '':
                    self.event_listener.fire('aborted')

                self.close()
                return False

            if 'registration_failed' == message['a']:
                if channel == '':
                    self.event_listener.fire('registration_failed', {'reason': message['reason']})
                self.close()
                return False

            if 'registered' == message['a']:
                self.registered[channel] = True
                if channel == '':
                    self.event_listener.fire('registration')

                self.handle_messages(channel, messages)
                return True

        self.logger.error("[%s] Registration of job %s failed." % (channel, self.job_id,))
        return False

    def handle_messages(self, channel, messages):
        BackendClient.handle_messages(self, channel, messages)

        # only main channel should handle following messages
        if channel != '':
            return

        for message in messages:
            if self.external_stopped:
                continue

            if not isinstance(message, dict):
                continue


            if 'a' in message and 'parameter-changed' == message['a']:
                self.event_listener.fire('parameter_changed', {'values': message['values']})

            if 'type' in message and 'action' == message['type']:
                self.event_listener.fire('action', message)
