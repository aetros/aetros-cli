from __future__ import absolute_import
from __future__ import print_function
import os
import pprint
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

        buffer = buffer[term_position + 1:]

    return buffer, parsed


class EventListener:
    def __init__(self):
        self.events = {}

    def on(self, name, callback):
        if name not in self.events:
            self.events[name] = []

        self.events[name].append(callback)

    def fire(self, name, parameter=None):
        if name in self.events:
            for callback in self.events[name]:
                callback(parameter)


class Client:
    def __init__(self, api_host, api_token, event_listener):
        """

        :type api_host: string
        :type api_token: string
        :type event_listener: EventListener
        :type job_id: integer
        """
        self.api_token = api_token
        self.api_host = api_host
        self.event_listener = event_listener
        self.message_id = 0
        self.job_id = None
        self.thread_instance = None
        self.s = None

        self.lock = Lock()
        self.connection_errors = 0
        self.queue = []

        self.active = False
        self.registered = False
        self.connected = False
        self.read_buffer = ''

    def start(self, job_id):
        self.job_id = job_id

        self.active = True

        if not self.thread_instance:
            self.thread_instance = Thread(target=self.thread)
            self.thread_instance.daemon = True
            self.thread_instance.start()

    def connect(self):
        locked = False

        try:
            locked = True
            self.lock.acquire()
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((self.api_host, 8051))
            self.connected = True
            self.lock.release()
            locked = False

            self.send_message({'register_job_worker': self.api_token, 'job_id': self.job_id})
            messages = self.read_full_message(self.s)

            pprint.pprint(messages)
            if "JOB_REGISTERED" in messages:
                self.registered = True
                print("Connected to %s " % (self.api_host,))
                self.handle_messages(messages)
                return True

            print("Registration of job %s failed." % (self.job_id,))
            return False
        except socket.error as error:
            if locked:
                self.lock.release()
            print("Connection error during connecting to %s: %d: %s." % (self.api_host, error.errno, error.message))
            time.sleep(1)
            return False

    def debug(self):
        sent = len(filter(lambda x: x['sent'], self.queue))
        sending = len(filter(lambda x: x['sending'], self.queue))
        open = len(filter(lambda x: not x['sending'], self.queue))
        print("%d sent, %d in sending, %d open " % (sent, sending, open))

    def connection_error(self, error=None):
        if error:
            print("Connection error: %d: %s" % (error.errno, error.message,))
        else:
            print("Connection error")

        self.connected = False
        self.registered = False

        # set all messages that are in sending to sending=false
        for message in self.queue:
            if message['sending'] and not message['sent']:
                message['sending'] = False

        self.connection_errors += 1

    def thread(self):
        last_ping = 0

        while True:
            if self.connected and self.registered:
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
                    self.connection_error(error)

            elif not self.connected and self.active:
                if not self.connect():
                    time.sleep(5)

            time.sleep(0.1)

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
        if isinstance(message, dict):
            message['sending'] = True

        msg = json.dumps(message, default=invalid_json_values)

        try:
            self.lock.acquire()
            self.s.sendall(msg + "\n")
            self.lock.release()

            return True
        except:
            self.lock.release()
            self.connection_error()
            return False

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

        :return: list
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


def create_job(name, api_token=None):
    """
    :param name: string
    :param api_token: string
    :return: JobBackend
    """
    job = JobBackend(api_token=api_token)
    job.create(name)

    return job


class JobChannel:
    """
    :type job_backend : JobBackend
    """

    def __init__(self, job_backend, name, series=None):
        self.name = name
        self.job_backend = job_backend

        if not series:
            series = [name]

        message = {
            'name': name,
            'series': series,
            'type': 'number'
        }
        self.job_backend.job_add_status('channel', message)

    def send(self, x, y):
        if not isinstance(y, list):
            y = [y]

        message = {
            'name': self.name,
            'x': x,
            'y': y
        }
        self.job_backend.job_add_status('channel-value', message)


class JobGraph:
    """
    :type job_backend : JobBackend
    """

    def __init__(self, job_backend, name, series=None):
        self.name = name
        self.job_backend = job_backend

        if not series:
            series = [name]

        message = {
            'name': name,
            'series': series,
            'type': 'number'
        }
        self.series = series
        self.job_backend.job_add_status('graph', message)

    def send(self, x, y):
        if not isinstance(y, list):
            y = [y]

        if len(y) != len(self.series):
            raise Exception('You tried to set more y values (%d items) then series available (%d series).' % (
            len(y), len(self.series)))

        message = {
            'name': self.name,
            'x': x,
            'y': y
        }
        self.job_backend.job_add_status('graph-value', message)


class JobBackend:
    """
    :type event_listener: EventListener
    :type api_token: string
    :type job_id: int
    :type client: Client
    :type job: dict
    """

    def __init__(self, job_id=None, api_token=None):
        self.event_listener = EventListener()
        self.api_token = api_token if api_token else os.getenv('API_KEY')

        self.job_id = job_id
        self.active_syncer = True
        self.client = None
        self.job = None

        self.host = os.getenv('API_HOST')
        if not self.host or self.host == 'false':
            self.host = 'aetros.com'

        self.job_ids = []
        self.in_request = False
        self.stop_requested = False
        self.event_listener.on('stop', self.external_stop)
        self.client = Client(self.host, self.api_token, self.event_listener)

    def external_stop(self):
        print("Job stopped through AETROS Trainer.")
        self.stop()
        self.stop_requested = True
        os.kill(os.getpid(), signal.SIGINT)

    def create_channel(self, name, series=None):
        """

        :param name: string
        :param series: list
        :return: JobChannel
        """
        return JobChannel(self, name, series)

    def create_graph(self, name, series=None):
        """

        :param name: string
        :param series: list
        :return: JobGraph
        """
        return JobGraph(self, name, series)

    def start(self):
        if not self.job_id:
            raise Exception('No job id found. Use create() first.')

        print("start client")
        self.client.start(self.job_id)

    def end(self):
        self.set_status('END')
        self.stop()

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

    def write_log(self, message):
        item = {'message': message}

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
        print('Job status changed to %s ' % (status,))
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

    def create(self, name, server_id='local', dataset_id=None, insights=False):
        response = self.put('job', {'modelId': name, 'serverId': server_id, 'insights': insights, 'datasetId': dataset_id})

        if response.status_code != 200:
            raise Exception("Could not create job: %s" % (response.content,))

        self.job_id = response.json()

        return self.job_id

    def ensure_model(self, name, model_json, settings=None, type='custom', layers=None, graph=None):
        response = self.put('model/ensure', {
            'id': name,
            'type': type,
            'model': model_json,
            'settings': json.dumps(settings, allow_nan=False, default=invalid_json_values) if settings else None,
            'layers': json.dumps(layers, allow_nan=False, default=invalid_json_values),
            'graph': json.dumps(graph, allow_nan=False, default=invalid_json_values),
        })

        if response.status_code != 200:
            raise Exception("Could not create model: %s" % (response.content,))

        return True

    @property
    def model_id(self):
        if not self.job:
            raise Exception('Job not loaded yet. Use load(id) first.')

        return self.job['modelId']

    def load(self, id):
        """
        Loads job and sets as current.
        :param id: int
        """
        if id:
            self.job_id = id

        if not self.job_id:
            raise Exception('No job id given.')

        response = self.get('job', {'id': self.job_id})

        if response.status_code != 200:
            raise Exception("Could not find job: %s" % (response.content,))

        job = response.json()

        if job is None or 'error' in job:
            raise Exception('Job not found. Have you configured your token correctly? %s: %s' %
                            (job['error'], job['message']))

        self.job = response.json()

    def get_job_model(self):
        """
        Returns a new JobModel instance with current loaded job data attached.
        :return: JobModel
        """
        if not self.job:
            raise Exception('Job not loaded yet. Use load(id) first.')

        from aetros.JobModel import JobModel

        return JobModel(self.job)

    def sync_weights(self):
        self.job_add_status('status', 'SYNC WEIGHTS')
        print("Sync weights ...")
        self.upload_weights('best.hdf5', self.get_job_model().get_weights_filepath_best(), with_status=True)
        print("Weights synced.")

    def load_light_job(self, id=None):
        """
        Loads job with less information and sets as current.
        :param id: int
        """
        if id:
            self.job_id = id

        if not self.job_id:
            raise Exception('No job id given.')

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

        self.job = job

    def job_add_status(self, key, value):
        item = {'statusKey': key,
                'statusValue': json.dumps(value, allow_nan=False, default=invalid_json_values)}

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
