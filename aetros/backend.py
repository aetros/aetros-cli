from __future__ import absolute_import
from __future__ import print_function

import atexit
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
import six
import base64
import PIL.Image
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

from aetros.MonitorThread import MonitoringThread


def on_shutdown():
    for job in on_shutdown.started_jobs:
        if job.running:
            job.done()


on_shutdown.started_jobs = []

atexit.register(on_shutdown)


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
    raise TypeError('Invalid data type passed to json encoder: ' + type(obj).__name__)


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
    def __init__(self, api_host, api_token, event_listener, api_port):
        """

        :type api_host: string
        :type api_token: string
        :type event_listener: EventListener
        :type job_id: integer
        """
        self.api_token = api_token
        self.api_host = api_host
        self.api_port = api_port
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
            self.s.connect((self.api_host, self.api_port))
            self.connected = True
            self.lock.release()
            locked = False

            self.send_message({'register_job_worker': self.api_token, 'job_id': self.job_id})
            messages = self.read_full_message(self.s)

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
        if not self.connected:
            return

        if socket is None:
            # python interpreter is already dying, so quit
            return

        if hasattr(error, 'errno') and hasattr(error, 'message'):
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

                except Exception as error:
                    self.connection_error(error)

            elif not self.connected and self.active:
                if not self.connect():
                    time.sleep(5)

            time.sleep(0.1)

    def end(self):
        #send all missing messages
        while True:
            if len(self.queue) == 0:
                break

            time.sleep(0.1)


    def close(self):
        self.active = False
        self.connected = False

        self.lock.acquire()
        try:
            self.s.shutdown(socket.SHUT_RDWR)
            self.s.close()
        except:
            pass
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


def start_job(name, api_token=None):
    """
    Tries to load the job defined in the AETROS_JOB_ID environment variable.
    If not defined, it creates a new job.
    Starts the job as well.

    :param name: string
    :param api_token: string
    :return: JobBackend
    """

    job_id = os.getenv('AETROS_JOB_ID')
    if job_id:
        job = JobBackend(api_token=api_token)
        job.load(job_id)
    else:
        job = create_job(name, api_token)

    job.start()

    return job


def create_job(name, api_token=None):
    """
    Creates a new job.

    :param name: string
    :param api_token: string
    :return: JobBackend
    """
    job = JobBackend(api_token=api_token)
    job.create(name)
    job.load()

    return job


class JobLossChannel:
    """
    :type job_backend : JobBackend
    """
    def __init__(self, job_backend, name, xaxis=None, yaxis=None, layout=None):
        self.name = name
        self.job_backend = job_backend
        message = {
            'name': self.name,
            'traces': ['training', 'validation'],
            'type': JobChannel.NUMBER,
            'main': True,
            'xaxis': xaxis,
            'yaxis': yaxis,
            'layout': layout,
            'lossChannel': True
        }
        self.job_backend.job_add_status('channel', message)

    def send(self, x, training_loss, validation_loss):

        message = {
            'name': self.name,
            'x': x,
            'y': [training_loss, validation_loss],
        }
        self.job_backend.job_add_status('channel-value', message)

class JobImage:
    def __init__(self, id, image, title=None):
        self.id = id
        if not isinstance(image, PIL.Image.Image):
            raise Exception('JobImage requires a PIL.Image as image argument.')

        self.image = image
        self.title = title or id


class JobChannel:
    NUMBER = 'number'
    TEXT = 'text'
    IMAGE = 'image'

    """
    :type job_backend: JobBackend
    """
    def __init__(self, job_backend, name, traces=None, main_graph=False,
                 type=None, xaxis=None, yaxis=None, layout=None):
        self.name = name
        self.job_backend = job_backend

        if not (isinstance(traces, list) or traces is None):
            raise Exception('traces can only be None or a list of dicts: [{name: "name", option1: ...}, ...]')

        if not traces:
            traces = [{'name': name}]

        if isinstance(traces, list) and isinstance(traces[0], six.string_types):
            traces = map(lambda x: {'name': x}, traces)

        message = {
            'name': name,
            'traces': traces,
            'type': type or JobChannel.NUMBER,
            'main': main_graph,
            'xaxis': xaxis,
            'yaxis': yaxis,
            'layout': layout,
        }
        self.traces = traces
        self.job_backend.job_add_status('channel', message)

    def send(self, x, y):
        if not isinstance(y, list):
            y = [y]

        if len(y) != len(self.traces):
            raise Exception('You tried to set more y values (%d items) then traces available in channel %s (%d traces).' % (
                len(y), self.name, len(self.traces)))

        message = {
            'name': self.name,
            'x': x,
            'y': y
        }
        self.job_backend.job_add_status('channel-value', message)


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
        self.client = None
        self.job = None
        self.running = False
        self.monitoring_thread = None

        self.host = os.getenv('API_HOST')
        self.port = int(os.getenv('API_PORT') or 8051)
        if not self.host or self.host == 'false':
            self.host = 'aetros.com'

        self.last_progress_call = None
        self.job_ids = []
        self.in_request = False
        self.stop_requested = False
        self.event_listener.on('stop', self.external_stop)
        self.client = Client(self.host, self.api_token, self.event_listener, self.port)

    def external_stop(self, params):
        print("Job stopped through AETROS Trainer.")
        self.abort()
        self.stop_requested = True
        os.kill(os.getpid(), signal.SIGINT)

    def progress(self, epoch, total):
        if self.last_progress_call:
            # how long took it since the last call?
            time_per_epoch = time.time() - self.last_progress_call
            eta = time_per_epoch * (total-epoch)
            self.set_info('eta', eta)
            if time_per_epoch > 0:
                self.set_info('epochsPerSecond', 1 / time_per_epoch)

        self.set_info('epoch', epoch)
        self.set_info('epochs', total)
        self.last_progress_call = time.time()

    def create_loss_channel(self, name, xaxis=None, yaxis=None, layout=None):
        """
        :param name: string
        :return: JobLossGraph
        """

        return JobLossChannel(self, name, xaxis, yaxis, layout)

    def create_channel(self, name, traces=None, main_graph=False,
                       type=JobChannel.NUMBER,
                       xaxis=None, yaxis=None, layout=None):
        """
        :param name: string
        :param traces: list
        :param main_graph: bool
        :param type: string JobChannel.NUMBER, JobChannel.TEXT, JobChannel.IMAGE
        :return: JobChannel
        """
        return JobChannel(self, name, traces, main_graph, type, xaxis, yaxis, layout)

    def start(self):
        if not self.job_id:
            raise Exception('No job id found. Use create() first.')

        self.running = True
        on_shutdown.started_jobs.append(self)
        self.collect_system_information()
        self.start_monitoring()

        self.client.start(self.job_id)

        print("Job %s#%d (%s) started. Open http://%s/trainer/app#/training=%s to monitor the training." %
              (self.model_id, self.job_index, self.job_id, self.host, self.job_id))

    def start_monitoring(self):
        self.monitoring_thread = MonitoringThread(self)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def done(self):
        if not self.running:
            return

        self.post('job/stopped', json={'id': self.job_id})

        self.client.end()
        self.running = False

        if self.monitoring_thread:
            self.monitoring_thread.stop()

    def abort(self):
        if not self.running:
            return

        self.post('job/aborted', json={'id': self.job_id})

        self.client.close()
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.stop()

    def crash(self, e=None):
        self.post('job/crashed', json={'id': self.job_id, 'error': e.message if e else None})

        self.client.close()
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.stop()

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

    def create(self, name, server_id='local', hyperparameter=None, dataset_id=None, insights=False):
        response = self.put('job', json={
            'modelId': name,
            'serverId': server_id,
            'hyperParameters': hyperparameter,
            'insights': insights,
            'datasetId': dataset_id
        })

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

    @property
    def job_index(self):
        if not self.job:
            raise Exception('Job not loaded yet. Use load(id) first.')

        return self.job['index']

    def is_keras_model(self):
        if not self.job:
            raise Exception('Job not loaded yet. Use load(id) first.')

        return not self.job['config']['fromCode']

    def get_parameter(self, name):
        if not self.job:
            raise Exception('Job not loaded yet. Use load(id) first.')

        if 'hyperParameters' not in self.job['config'] or not self.job['config']['hyperParameters']:
            raise Exception('This job does not have any hyper-parameters')

        if name not in self.job['config']['hyperParameters']:
            raise Exception('This job does not have the hype parameter %s' % (name,))

        return self.job['config']['hyperParameters'][name]

    def load(self, id=None):
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
        item = {'statusKey': key, 'statusValue': value}

        self.client.send({
            'type': 'job-status',
            'time': time.time(),
            'data': item
        })

    def set_info(self, name, value):
        self.job_set_info_key(name, value)

    def job_set_info_key(self, key, value):
        self.client.send({
            'type': 'job-info',
            'data': {'key': key, 'value': value}
        })

    def job_add_insight(self, x, images, confusion_matrix):
        info = {'epoch': x, 'confusionMatrix': confusion_matrix}

        converted_images = []
        for image in images:
            if not isinstance(image, JobImage):
                raise Exception('job_add_insight only accepts JobImage instances in images argument')

            converted_images.append({
                'id': image.id,
                'title': image.title,
                'image': self.to_base64(image.image)
            })

        self.client.send({
            'type': 'job-insight',
            'time': time.time(),
            'data': {'info': info, 'images': converted_images}
        })

    def to_base64(self, image):
        buffer = BytesIO()
        if (six.PY2):
            buffer = StringIO()
        image.save(buffer, format="JPEG", optimize=True, quality=80)
        return base64.b64encode(buffer.getvalue())


    def collect_system_information(self):
        import psutil

        mem = psutil.virtual_memory()
        self.job_set_info_key('memory_total', mem.total)

        on_gpu = False

        import sys
        if 'theano.sandbox' in sys.modules:
            # at this point, theano is already initialised, so we can use it to monitor the GPU.
            from theano.sandbox import cuda
            self.job_set_info_key('cuda_available', cuda.cuda_available)
            if cuda.cuda_available:
                on_gpu = cuda.use.device_number is not None
                self.job_set_info_key('cuda_device_number', cuda.active_device_number())
                self.job_set_info_key('cuda_device_name', cuda.active_device_name())
                if cuda.cuda_ndarray.cuda_ndarray.mem_info:
                    gpu = cuda.cuda_ndarray.cuda_ndarray.mem_info()
                    self.job_set_info_key('cuda_device_max_memory', gpu[1])
                    free = gpu[0] / 1024 / 1024 / 1024
                    total = gpu[1] / 1024 / 1024 / 1024
                    used = total - free

                    print("%.2fGB GPU memory used of %.2fGB, %s, device id %d" % (used, total, cuda.active_device_name(), cuda.active_device_number()))

        self.job_set_info_key('on_gpu', on_gpu)

        import cpuinfo
        cpu = cpuinfo.get_cpu_info()
        self.job_set_info_key('cpu_name', cpu['brand'])
        self.job_set_info_key('cpu', [cpu['hz_actual_raw'][0], cpu['count']])
