from __future__ import absolute_import
from __future__ import print_function
import os
import socket
import select
from threading import Thread, Lock

import requests

import signal
import json
import time
import numpy

from io import BytesIO


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def invalid_json_values(obj):
    if isinstance(obj, numpy.generic):
        return obj.item()
    if isinstance(obj, numpy.ndarray):
        return obj.tolist()
    if isinstance(obj, bytes):
        return obj.decode('cp437')
    raise TypeError('Invalid data type passed to json encoder: ' + type(obj))


def parse_message(buffer):
    parsed = []
    while -1 != buffer.find('\n'):
        term_position = buffer.find('\n')
        messages = buffer[0:term_position]
        messages = messages.split('\t')
        for message in messages:
            if message:
                parsed.append(json.loads(message))

        buffer = buffer[term_position+1:]

    return buffer, parsed


class EventListener:
    def __init__(self):
        self.events = {}

    def on(self, name, callback):
        if name not in self.events:
            self.events[name] = []

        self.events[name].append(callback)

    def fire(self, name):
        if name in self.events:
            for callback in self.events[name]:
                callback()


class Client:
    def __init__(self, api_host, api_token, event_listener, job_id):
        """

        :type api_host: string
        :type api_token: string
        :type event_listener: EventListener
        :type job_id: integer
        """

        self.api_token = api_token
        self.api_host = api_host
        self.event_listener = event_listener
        self.job_id = job_id
        self.message_id = 0

        self.lock = Lock()
        self.connection_errors = 0
        self.queue = []

        self.active = True
        self.authenticated = False
        self.job_registered = False
        self.connected = False
        self.read_buffer = ''

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

                self.send_message({'register_job': self.job_id})
                messages = self.read_full_message(self.s)

                if "JOB_REGISTERED" in messages:
                    self.job_registered = True
                else:
                    print("Registration of job %s failed." % (self.job_id,))
                    return

                break
            except socket.error as error:
                self.lock.release()
                print("Connection error during connecting to %s: %d: %s." % (self.api_host, error.errno, error.message))
                time.sleep(1)

        print("Connected to %s " % (self.api_host,))

    def debug(self):
        sent = len(filter(lambda x: x['sent'], self.queue))
        sending = len(filter(lambda x: x['sending'], self.queue))
        open = len(filter(lambda x: not x['sending'], self.queue))
        print("%d sent, %d in sending, %d open " % (sent, sending, open))

    def connection_error(self):
        self.connected = False
        self.authenticated = False
        self.job_registered = False

        # set all messages that are in sending to sending=false
        for message in self.queue:
            if message['sending'] and not message['sent']:
                message['sending'] = False

        self.connection_errors += 1
        self.connect()

    def thread(self):
        last_ping = 0

        while True:
            if self.connected and self.authenticated and self.job_registered:
                try:
                    if last_ping < time.time() - 1:
                        # ping every second
                        last_ping = time.time()
                        self.send_message("PING")

                    if self.connected:
                        # send pending messages
                        max_messages = 1
                        for message in self.queue:
                            if not message['sending'] and not message['sent']:
                                max_messages += 1
                                self.send_message(message)
                                if max_messages > 10:
                                    # not too much at once, so we have time to listen for incoming messages
                                    break

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
                    self.connected = False
                    print("Connection error: %d: %s" % (error.errno, error.message,))
                    self.connection_error()

    def close(self):
        self.active = False
        self.connected = False

        self.lock.acquire()
        self.s.shutdown(socket.SHUT_RDWR)
        self.s.close()
        self.lock.release()

    def send(self, message):
        self.message_id += 1
        message['id'] = self.message_id
        message['sending'] = False
        message['sent'] = False
        self.queue.append(message)

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

    def handle_messages(self, messages):
        for message in messages:
            if 'handled' in message:
                for qm in self.queue:
                    if qm['id'] == message['id']:

                        if message['handled']:
                            qm['sent'] = True
                            self.queue.remove(qm)
                        else:
                            qm['sending'] = False

                        break

            if 'stop' in message:
                self.event_listener.fire('stop')

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


class AetrosBackend:
    def __init__(self, job_id=None):
        self.event_listener = EventListener()
        self.api_token = os.getenv('API_KEY')

        self.job_id = job_id
        self.active_syncer = True

        self.host = os.getenv('API_HOST')
        if not self.host or self.host == 'false':
            self.host = 'aetros.com'

        self.job_ids = []
        self.in_request = False
        self.stop_requested = False
        self.event_listener.on('stop', self.external_stop)

    def external_stop(self):
        print("Job stopped through AETROS Trainer.")
        self.stop()
        self.stop_requested = True
        os.kill(os.getpid(), signal.SIGINT)

    def start(self, id):
        self.client = Client(self.host, self.api_token, self.event_listener, id)

    def stop(self):
        self.client.close()

    def kill(self):
        self.active_syncer = False

    def get_url(self, affix):

        url = 'http://%s/api/%s' % (self.host, affix)

        if self.api_token:
            if '?' in url:
                url += '&token=' + self.api_token
            else:
                url += '?token=' + self.api_token

        return url

    def write_log(self, id, message):
        item = {'id': id, 'message': message}

        self.client.send({
            'type': 'log',
            'time': time.time(),
            'data': item
        })

    def upload_weights(self, name, file_path, accuracy=None, with_status=False):
        if not os.path.isfile(file_path):
            return

        files = {}
        files[name] = open(file_path, 'r')

        class CancelledError(Exception):

            def __init__(self, msg):
                self.msg = msg
                Exception.__init__(self, msg)

            def __str__(self):
                return self.msg

            __repr__ = __str__

        class BufferReader(BytesIO):

            def __init__(self, buf=b'',
                         callback=None,
                         cb_args=(),
                         cb_kwargs={}):
                self._callback = callback
                self._cb_args = cb_args
                self._cb_kwargs = cb_kwargs
                self._progress = 0
                self._len = len(buf)
                BytesIO.__init__(self, buf)

            def __len__(self):
                return self._len

            def read(self, n=-1):
                chunk = BytesIO.read(self, n)
                self._progress += int(len(chunk))
                self._cb_kwargs.update({
                    'size': self._len,
                    'progress': self._progress
                })
                if self._callback:
                    try:
                        self._callback(*self._cb_args, **self._cb_kwargs)
                    except:
                        raise CancelledError('The upload was cancelled.')
                return chunk

        state = {'progress': -1}

        def progress(size=None, progress=None):
            current_progress = int(progress / size * 100)
            if state['progress'] != current_progress:
                state['progress'] = current_progress

                if with_status:
                    self.set_status('UPLOAD WEIGHTS ' + str(current_progress) + '%')
                else:
                    print(("{0}%".format(current_progress)))

        files = {"upfile": (name, open(file_path, 'rb').read())}

        (data, ctype) = requests.packages.urllib3.filepost.encode_multipart_formdata(files)

        headers = {
            "Content-Type": ctype
        }

        body = BufferReader(data, progress)

        url = self.get_url('job/weights?id=%s&accuracy=%f.2' %
                           (self.job_id, accuracy if accuracy is not None else -1))
        response = requests.post(url, data=body, headers=headers)

        if response.status_code != 200:
            raise Exception('Uploading of weights failed: %d: %s' %
                            (response.status_code, response.content))

    def set_status(self, status):
        print('Training status changed to %s ' % (status,))
        self.job_add_status('status', status)

    def get_best_weight_url(self, job_id):
        response = self.get('job/weight-best', {'id': job_id})
        return response.json()

    def get(self, url, params=None, **kwargs):
        json_chunk = kwargs.get('json')
        if (json_chunk and not isinstance(json_chunk, str)):
            kwargs['json'] = json.loads(json.dumps(json_chunk, default=invalid_json_values))

        return requests.get(self.get_url(url), params=params, **kwargs)

    def post(self, url, data=None, **kwargs):
        json_chunk = kwargs.get('json')
        if (json_chunk and not isinstance(json_chunk, str)):
            kwargs['json'] = json.loads(json.dumps(json_chunk, default=invalid_json_values))

        return requests.post(self.get_url(url), data=data, **kwargs)

    def put(self, url, data=None, **kwargs):
        json_chunk = kwargs.get('json')
        if (json_chunk and not isinstance(json_chunk, str)):
            kwargs['json'] = json.loads(json.dumps(json_chunk, default=invalid_json_values))

        return requests.put(self.get_url(url), data=data, **kwargs)

    def create_job(self, name, server_id='local', dataset_id=None, insights=False):
        response = self.put('job',
                            {'networkId': name, 'serverId': server_id, 'insights': insights, 'datasetId': dataset_id})
        if response.status_code != 200:
            raise Exception("Could not create training: %s" % (response.content,))

        job_id = response.json()
        self.job_id = job_id

        return job_id

    def ensure_network(self, network_name, model_json, settings=None, network_type='custom', layers=None, graph=None):
        response = self.put('network/ensure', {
            'id': network_name,
            'type': network_type,
            'model': model_json,
            'settings': json.dumps(settings, allow_nan=False, default=invalid_json_values) if settings else None,
            'layers': json.dumps(layers, allow_nan=False, default=invalid_json_values),
            'graph': json.dumps(graph, allow_nan=False, default=invalid_json_values),
        })

        if response.status_code != 200:
            raise Exception("Could not create network: %s" % (response.content,))

        return True

    def get_job(self):
        response = self.get('job', {'id': self.job_id})
        if response.status_code != 200:
            raise Exception("Could not find version: %s" % (response.content,))

        job = response.json()

        if job is None or 'error' in job:
            raise Exception('Training/Version not found. Have you configured your token correctly? %s: %s' %
                            (job['error'], job['message']))

        return response.json()

    def get_light_job(self):
        response = self.get('job', {'id': self.job_id, 'light': 1})
        if response.status_code != 200:
            raise Exception("Could not find version (%s): %s" % (self.job_id, response.content,))

        job = response.json()

        if job is None or job == 'Job not found':
            raise Exception('Version not found. Have you configured your token correctly?')

        if 'error' in job:
            raise Exception('Version not found. Have you configured your token correctly? %s: %s' % (
                job['error'], job['message']))

        if not isinstance(job, dict):
            raise Exception(
                'Version does not exist. Make sure you created the job via AETROS TRAINER')

        if not len(job['config']):
            raise Exception(
                'Version does not have a configuration. Make sure you created the job via AETROS TRAINER')

        return job

    def job_started(self, id, pid):
        self.start(id)
        return self.post('job/started', {'id': id, 'pid': pid})

    def job_add_status(self, statusKey, statusValue):
        item = {'statusKey': statusKey,
                'statusValue': json.dumps(statusValue, allow_nan=False, default=invalid_json_values)}

        self.client.send({
            'type': 'job-status',
            'time': time.time(),
            'data': item
        })

    def job_set_info_key(self, key, info):
        self.client.send({
            'type': 'job-info',
            'time': time.time(),
            'data': {'key': key, 'info': info}
        })

    def job_add_insight(self, info, images):
        self.client.send({
            'type': 'job-insight',
            'time': time.time(),
            'data': {'info': info, 'images': images}
        })
