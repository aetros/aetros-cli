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
import sys
import msgpack

from aetros.const import JOB_STATUS
from aetros.logger import GeneralLogger
from aetros.utils import git

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

from aetros.MonitorThread import MonitoringThread


def on_shutdown():
    for job in on_shutdown.started_jobs:
        job.on_shutdown()


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
    if isinstance(obj, map):
        return list(obj)
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


class BackendClient:
    def __init__(self, api_host, api_port, event_listener):
        """

        :type api_host: string
        :type api_key: string
        :type event_listener: EventListener
        :type job_id: integer
        """
        self.api_host = api_host or os.getenv('API_HOST') or 'trainer.aetros.com'
        self.api_port = int(api_port or os.getenv('API_PORT') or 8051)

        self.event_listener = event_listener
        self.message_id = 0

        self.api_key = None
        self.job_id = None

        self.thread_instance = None
        self.s = None

        self.lock = Lock()
        self.connection_errors = 0
        self.queue = []

        self.active = False
        self.external_stopped = False
        self.registered = False
        self.connected = False
        self.read_unpacker = msgpack.Unpacker(encoding='utf8')

    def start(self):
        self.active = True

        if not self.thread_instance:
            self.thread_instance = Thread(target=self.thread)
            self.thread_instance.daemon = True
            self.thread_instance.start()

    def on_connect(self):
        pass

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

            return self.on_connect()
        except socket.error as error:
            if locked:
                self.lock.release()
            if hasattr(error, 'message'):
                print("Connection error during connecting to %s: %d: %s." % (self.api_host, error.errno, error.message))
            else:
                print("Connection error during connecting to %s." % (self.api_host,))
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
        elif error:
            print("Connection error: " + str(error))
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
                if last_ping < time.time() - 1:
                    # ping every second
                    last_ping = time.time()
                    self.send_message("PING")

                # see if we can read something
                self.lock.acquire()
                readable, writable, exceptional = select.select([self.s], [self.s], [self.s])
                self.lock.release()

                if exceptional:
                    self.connection_error()

                if readable:
                    messages = self.read(self.s)
                    if messages is not None:
                        self.handle_messages(messages)

                if writable:
                    # send pending messages
                    self.lock.acquire()
                    sent_size = 0
                    for message in self.queue:
                        if not message['sending'] and not message['sent']:
                            size = self.send_message(message)
                            if size is not False:
                                sent_size += size
                                if sent_size > 1024 * 1024:
                                    # not too much at once (max 1MB), so we have time to listen for incoming messages
                                    break
                    self.lock.release()

            elif not self.connected and self.active:
                if not self.connect():
                    time.sleep(5)

            time.sleep(0.1)

    def end(self):
        # send all missing messages

        i = 0
        while True:
            if len(self.queue) == 0:
                break

            i += 1
            time.sleep(0.1)
            if i % 50 == 0:
                print("[AETROS] we still sync job data. %d messages left." % (len(self.queue),))

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

        import msgpack
        msg = msgpack.packb(message, default=invalid_json_values)

        try:
            self.s.sendall(msg)

            return len(msg)
        except:
            self.connection_error()
            return False

    def handle_messages(self, messages):
        for message in messages:
            if 'handled' in message and message['handled'] is True:
                for qm in self.queue:
                    if qm['id'] == message['id']:

                        if message['handled']:
                            qm['sent'] = True
                            self.queue.remove(qm)
                        else:
                            qm['sending'] = False

                        break

            if not self.external_stopped and 'stop' in message:
                self.external_stopped = True
                self.event_listener.fire('stop')

    def wait_for_at_least_one_message(self, s):
        """
        Reads until we receive at least one message we can unpack. Return all found messages.
        """

        unpacker = msgpack.Unpacker(encoding='utf8')

        while True:
            chunk = ''
            try:
                self.lock.acquire()
                chunk = s.recv(1024)
            except:
                self.connection_error()
                return None
            finally:
                self.lock.release()

            if chunk == '':
                # happens only when connection broke. If nothing is to be received, it hangs instead.
                self.connection_error()
                return False

            unpacker.feed(chunk)

            messages = [m for m in unpacker]
            if messages:
                return messages

    def read(self, s):
        """
        Reads from the socket and tries to unpack the message. If successful (because msgpack was able to unpack)
        then we return that message. Else None. Keep calling .read() when new data is available so we try it 
        again.
        """

        self.lock.acquire()
        try:
            chunk = s.recv(1024 * 1024)
        except:
            self.connection_error()
            return None
        finally:
            self.lock.release()

        if chunk == '':
            # socket connection broken
            self.connection_error()
            return None

        # self.read_buffer.seek(0, 2) #make sure we write at the end
        self.read_unpacker.feed(chunk)

        # self.read_buffer.seek(0)
        messages = [m for m in self.read_unpacker]

        return messages if messages else None


class JobClient(BackendClient):
    def configure(self, job_id, api_key):
        self.job_id = job_id
        self.api_key = api_key

    def on_connect(self):
        self.send_message({'register_job_worker': self.api_key, 'job_id': self.job_id})
        messages = self.wait_for_at_least_one_message(self.s)

        message = messages.pop(0)
        if isinstance(message, dict) and 'a' in message:
            if "JOB_ABORTED" in message['a']:
                print("Job aborted meanwhile. Exiting")
                self.event_listener.fire('stop')
                self.active = False
                return False

            if "JOB_REGISTRATION_FAILED" in message['a']:
                self.event_listener.fire('registration_failed', {'reason': message['reason']})
                return False

            if "JOB_REGISTERED" in message['a']:
                self.registered = True
                print("Connected to %s " % (self.api_host,))
                self.event_listener.fire('registration')
                self.handle_messages(messages)
                return True

        print("Registration of job %s failed." % (self.job_id,))
        return False


def start_job(name, api_key=None):
    """
    Tries to load the job defined in the AETROS_JOB_ID environment variable.
    If not defined, it creates a new job.
    Starts the job as well.

    :param name: string: model name
    :param api_key: string
    :return: JobBackend
    """

    id = os.getenv('AETROS_JOB_ID')
    if id:
        job = JobBackend(api_key=api_key)
        job.load(id)
    else:
        job = create_job(name, api_key)

    job.start()
    job.setup_std_output_logging()

    return job


def create_job(name, api_key=None):
    """
    Creates a new job.

    :param name: string : model name
    :param api_key: string
    :return: JobBackend
    """
    job = JobBackend(api_key=api_key)
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

    def __init__(self, job_backend, name, traces=None,
                 main=False, kpi=False, max_optimization=True,
                 type=None, xaxis=None, yaxis=None, layout=None):
        """
        :param job_backend: JobBakend
        :param name: str
        :param traces: None|list : per default create a trace based on "name".
        :param main: bool : whether this channel is visible in the job list as column for better comparison.

        :param kpi: bool : whether this channel is the KPI (key performance indicator).
                           Used for hyperparameter optimization. Only one channel can be a kpi. Only first trace used.

        :param max_optimization: bool : whether the optimization maximizes or minmizes the kpi . Use max_optimization=False to
                                        tell the optimization algorithm that his channel minimizes a kpi, for instance the loss of a model.

        :param type: str : One of JobChannel.NUMBER, JobChannel.TEXT, JobChannel.IMAGE
        :param xaxis: dict
        :param yaxis: dict
        :param layout: dict
        """
        self.name = name
        self.job_backend = job_backend

        if not (isinstance(traces, list) or traces is None):
            raise Exception(
                'traces can only be None or a list of dicts: [{name: "name", option1: ...}, {name: "name2"}, ...]')

        if not traces:
            traces = [{'name': name}]

        if isinstance(traces, list) and isinstance(traces[0], six.string_types):
            traces = list(map(lambda x: {'name': x}, traces))

        message = {
            'name': name,
            'traces': traces,
            'type': type or JobChannel.NUMBER,
            'main': main,
            'kpi': kpi,
            'maxOptimization': max_optimization,
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
            raise Exception(
                'You tried to set more y values (%d items) then traces available in channel %s (%d traces).' % (
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
    :type api_key: string
    :type job_id: int
    :type client: Client
    :type job: dict
    """

    def __init__(self, job_id=None, api_key=None):
        self.event_listener = EventListener()
        self.api_key = api_key if api_key else os.getenv('API_KEY')

        if job_id and '/' in job_id:
            raise Exception('job_id needs to be a job id, not a model name.')

        on_shutdown.started_jobs.append(self)

        self.job_id = job_id
        self.client = None
        self.job = None

        self.last_batch_time = time.time()
        self.start_time = time.time()
        self.current_epoch = 0
        self.current_batch = 0
        self.total_epochs = 0
        self.made_batches = 0
        self.batches_per_second = 0

        # done means: done, abort or crash method has been called.
        self.ended = False

        # running means: the syncer client is running.
        self.running = False
        self.monitoring_thread = None
        self.general_logger_stdout = GeneralLogger(job_backend=self)
        self.general_logger_error = GeneralLogger(job_backend=self, error=True)

        self.host = os.getenv('API_HOST')
        self.port = int(os.getenv('API_PORT') or 8051)
        if not self.host or self.host == 'false':
            self.host = 'trainer.aetros.com'

        self.last_progress_call = None
        self.job_ids = []
        self.in_request = False
        self.stop_requested = False
        self.event_listener.on('stop', self.external_stop)
        self.event_listener.on('aborted', self.external_aborted)
        self.event_listener.on('registration', self.on_registration)
        self.event_listener.on('registration_failed', self.on_registration_failed)

        self.client = JobClient(self.host, self.port, self.event_listener)

    def on_registration_failed(self, params):
        print("Connecting to AETROS failed. Reasons: %s. Wrong API_KEY='%s'?" % (params['reason'], self.api_key,))
        print("All monitoring disabled for AETROS. Job may still run.", file=sys.stderr)
        self.crash('Client could not connect to AETROS. Job may still run. Reason: %s' % (params['reason'],))

    def on_registration(self, params):
        print("Job %s#%d (%s) started. Open http://%s/job/%s to monitor the training." %
              (self.model_id, self.job_index, self.job_id, self.host, self.job_id))

    def external_aborted(self, params):
        self.ended = True
        self.running = False

        if self.monitoring_thread:
            self.monitoring_thread.stop()

        self.client.close()

        os.kill(os.getpid(), signal.SIGINT)

    def setup_std_output_logging(self):
        """
        Overwrites sys.stdout and sys.stderr so we can send it additionally to Aetros.
        """
        sys.stdout = self.general_logger_stdout
        sys.stderr = self.general_logger_error

    def external_stop(self, params):
        print("Job stopped through AETROS Trainer.")
        self.abort()
        self.stop_requested = True
        os.kill(os.getpid(), signal.SIGINT)

    def batch(self, batch, total, size=None):
        time_diff = time.time() - self.last_batch_time
        self.made_batches += 1

        if time_diff > 1 or batch == total:  # only each second second or last batch
            self.set_system_info('currentBatch', batch)
            self.set_system_info('batchSize', size)
            self.set_system_info('nb_batches', total)

            self.batches_per_second = self.made_batches / time_diff
            self.made_batches = 0
            self.last_batch_time = time.time()

            if size:
                self.set_system_info('samplesPerSecond', self.batches_per_second * size)

            epochs_per_second = self.batches_per_second / total  # all batches
            self.set_system_info('epochsPerSecond', epochs_per_second)

            elapsed = time.time() - self.start_time
            self.set_system_info('elapsed', elapsed)

            if self.total_epochs:
                eta = 0
                if batch < total:
                    # time to end this epoch
                    eta += (total - batch) / self.batches_per_second

                # time until all epochs are done
                if self.total_epochs - (self.current_epoch - 1) > 0:
                    eta += (self.total_epochs - (self.current_epoch - 1)) / epochs_per_second

                self.set_system_info('eta', eta)

        self.current_batch = batch

    def progress(self, epoch, total):
        self.current_epoch = epoch
        self.total_epochs = total
        epoch_limit = False

        if 'maxEpochs' in self.job['config']['settings'] and self.job['config']['settings']['maxEpochs'] > 0:
            epoch_limit = True
            self.total_epochs = self.job['config']['settings']['maxEpochs']

        if epoch is not 0 and self.last_progress_call:
            # how long took it since the last call?
            time_per_epoch = time.time() - self.last_progress_call
            eta = time_per_epoch * (self.total_epochs - epoch)
            self.set_system_info('eta', eta)
            if time_per_epoch > 0:
                self.set_system_info('epochsPerSecond', 1 / time_per_epoch)

        self.set_system_info('epoch', epoch)
        self.set_system_info('epochs', self.total_epochs)
        self.last_progress_call = time.time()

        if epoch_limit and self.total_epochs > 0:
            if epoch >= self.total_epochs:
                print("\nMaxEpochs of %d/%d reached" % (epoch, self.total_epochs))
                self.done()
                os.kill(os.getpid(), signal.SIGINT)
                return

    def create_loss_channel(self, name, xaxis=None, yaxis=None, layout=None):
        """
        :param name: string
        :return: JobLossGraph
        """

        return JobLossChannel(self, name, xaxis, yaxis, layout)

    def create_channel(self, name, traces=None,
                       main=False, kpi=False, max_optimization=True,
                       type=JobChannel.NUMBER,
                       xaxis=None, yaxis=None, layout=None):
        """
        :param name: str
        :param traces: None|list : per default create a trace based on "name".
        :param main: bool : whether this channel is visible in the job list as column for better comparison.

        :param kpi: bool : whether this channel is the KPI (key performance indicator).
                           Used for hyperparameter optimization. Only one channel can be a kpi. Only first trace used.

        :param max_optimization: bool : whether the optimization maximizes or minmizes the kpi . Use max_optimization=False to
                                        tell the optimization algorithm that his channel minimizes a kpi, for instance the loss of a model.

        :param type: str : One of JobChannel.NUMBER, JobChannel.TEXT, JobChannel.IMAGE
        :param xaxis: dict
        :param yaxis: dict
        :param layout: dict
        """
        return JobChannel(self, name, traces, main, kpi, max_optimization, type, xaxis, yaxis, layout)

    def start(self):
        if self.running:
            raise Exception('Job already running.')

        if not self.job_id:
            raise Exception('No job id found. Use create() first.')

        self.running = True
        self.ended = False
        self.collect_system_information()
        self.collect_environment()
        self.start_monitoring()

        self.client.configure(self.job_id, self.api_key)
        self.client.start()
        self.detect_git_version()

    def detect_git_version(self):
        try:
            commit_sha = git.get_current_commit_hash()
            if commit_sha:
                self.set_system_info('gitVersion', commit_sha)

            current_branch = git.get_current_branch()
            if current_branch:
                self.set_system_info('gitBranch', current_branch)
        except:
            pass

    def start_monitoring(self):
        if not self.monitoring_thread:
            self.monitoring_thread = MonitoringThread(self)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()

    def create_keras_callback(self, model,
                              insights=False, insights_x=None,
                              additional_insights_layer=[],
                              confusion_matrix=False, validation_data=None, validation_data_size=None):

        """

        :type validation_data: int|None: (x, y) or generator
        :type validation_data_size: int|None: Defines the size of validation_data, if validation_data is a generator
        """
        self.ensure_model(self.model_id, settings={}, type='custom')

        from aetros.Trainer import Trainer
        self.trainer = Trainer(self)

        self.trainer.model = model

        from aetros.KerasCallback import KerasCallback
        self.callback = KerasCallback(self, self.general_logger_stdout, force_insights=insights)
        self.callback.insights_x = insights_x
        self.callback.insight_layer = additional_insights_layer
        self.callback.confusion_matrix = confusion_matrix
        self.callback.set_validation_data(validation_data, validation_data_size)

        return self.callback

    def upload_keras_graph(self, model):
        from aetros.keras import model_to_graph
        import keras

        if keras.__version__[0] == '2':
            graph = model_to_graph(model)
            self.set_graph(graph)

    def on_shutdown(self):
        if not self.ended and hasattr(sys, 'last_value'):
            # sys.last_value contains a exception, when there was an uncaught one
            if isinstance(sys.last_value, KeyboardInterrupt):
                self.abort()
            else:
                self.crash(type(sys.last_value).__name__ + ': ' + str(sys.last_value))

        elif self.running:
            self.done()

    def done(self):
        if not self.running:
            return

        if self.total_epochs:
            self.progress(self.total_epochs, self.total_epochs)

        self.post('job/stopped', json={'id': self.job_id})

        self.stop(True)

    def stop(self, wait_for_client=False):
        if self.monitoring_thread:
            self.monitoring_thread.stop()

        self.ended = True
        self.running = False

        if wait_for_client:
            self.client.end()
        else:
            self.client.close()

    def abort(self):
        if not self.running:
            return

        self.post('job/aborted', json={'id': self.job_id})

        self.stop()

    def crash(self, error=None):
        data = {'id': self.job_id, 'error': str(error) if error else None,
                'stderr': self.general_logger_error.last_messages}
        self.post('job/crashed', json=data)

        self.stop()

    def get_url(self, affix):

        url = 'http://%s/api/%s' % (self.host, affix)

        if self.api_key:
            if '?' in url:
                url += '&token=' + self.api_key
            else:
                url += '?token=' + self.api_key

        return url

    def write_log(self, message):
        item = {'message': message}

        self.client.send({
            'type': 'log',
            'time': time.time(),
            'data': item
        })

    def upload_weights(self, name, file_path, kpi=None, with_status=False):
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

        url = 'job/weights?id=%s' % (self.job_id, )

        if kpi is not None:
            url += '&kpi=' + str(kpi)

        url = self.get_url(url)
        print(url)
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

    def ensure_job(self, id, hyperparameter=None, dataset_id=None, server_id='local', insights=False,
                   insights_sample_path=None, api_key=None):
        """
        Makes sure that either
        1. id is a model name, a job is created and loaded, or
        2. id is a job id and job is loaded.

        :param id: integer|str: model name or job id.
        """

        if '/' in id:
            # it's a model name
            id = self.create(id, server_id=server_id, hyperparameter=hyperparameter,
                             dataset_id=dataset_id, insights=insights)
            self.load(id)

            print(
                "Job %s#%d (%s) created and started. Open http://%s/job/%s to monitor the training." %
                (self.model_id, self.job_index, self.job_id, self.host, id))
        else:
            self.load(id)
            if self.job['progressStatus'] > JOB_STATUS.PROGRESS_STATUS_QUEUED:
                self.restart(id)
                self.load(id)

    def ensure_model(self, name, settings=None, type='custom', layers=None):
        response = self.put('model/ensure', {
            'id': name,
            'type': type,
            'settings': json.dumps(settings, allow_nan=False, default=invalid_json_values) if settings else None,
            'layers': json.dumps(layers, allow_nan=False, default=invalid_json_values),
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

        params = self.job['config']['hyperParameters']
        if name in params:
            return params[name]

        if '.' in name:
            path = name.split('.')
            current = params
            for item in path:
                current = current[item]

            return current

        raise Exception('This job does not have the hype parameter %s' % (name,))

    def restart(self, id=None):
        """
        Restarts job.
        :param id: int
        """
        if id:
            self.job_id = id

        if not self.job_id:
            raise Exception('No job id given.')

        response = self.post('job/restart', {'id': self.job_id})

        if response.status_code != 200:
            raise Exception("Could not restart job: %s" % (response.content,))

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
        self.job_id = self.job['id']

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
        try:
            self.upload_weights('latest.hdf5', self.get_job_model().get_weights_filepath_latest(), with_status=True)
        except:
            pass
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

    def set_info(self, key, value):
        self.client.send({
            'type': 'job-custom-info',
            'data': {'key': key, 'value': value}
        })

    def set_graph(self, graph):
        self.client.send({
            'type': 'job-graph',
            'data': {'graph': graph}
        })

    def set_system_info(self, key, value):
        self.client.send({
            'type': 'job-info',
            'data': {'key': key, 'value': value}
        })

    def upload_file(self, file_path, name=None, mime_type=None):
        if os.path.getsize(file_path) > 10 * 1024 * 1024 * 1024:
            raise Exception('Can not upload file bigger than 10MB')

        with open(file_path, 'rb') as f:
            contents = f.read()

        if mime_type is None:
            import mimetypes
            mime_type, encoding = mimetypes.guess_type(file_path)

        path = os.path.relpath(file_path)

        self.client.send({
            'type': 'job-file',
            'time': time.time(),
            'data': {'path': path, 'name': name, 'mime': mime_type, 'content': contents}
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
                'image': self.pil_image_to_jpeg(image.image)
            })

        self.client.send({
            'type': 'job-insight',
            'time': time.time(),
            'data': {'info': info, 'images': converted_images}
        })

    def pil_image_to_jpeg(self, image):
        buffer = six.BytesIO()

        image.save(buffer, format="JPEG", optimize=True, quality=80)
        return buffer.getvalue()

    def collect_environment(self):
        import socket
        import os
        import pip
        import platform

        env = {}

        import aetros
        env['aetros_version'] = aetros.__version__
        env['python_version'] = platform.python_version()
        env['python_executable'] = sys.executable

        env['hostname'] = socket.gethostname()
        env['variables'] = dict(os.environ)

        if 'API_KEY' in env['variables']:
            del env['variables']['API_KEY']

        env['pip_packages'] = sorted([[i.key, i.version] for i in pip.get_installed_distributions()])
        self.set_system_info('environment', env)

    def collect_system_information(self):
        import psutil

        mem = psutil.virtual_memory()
        self.set_system_info('memory_total', mem.total)

        on_gpu = False

        import aetros.cuda_gpu
        props = aetros.cuda_gpu.get_device_properties(0)
        if props:
            self.set_system_info('cuda_available', True)
            self.set_system_info('cuda_device_number', props['device'])
            self.set_system_info('cuda_device_name', props['name'])
            self.set_system_info('cuda_device_max_memory', props['memory'])


        self.set_system_info('on_gpu', on_gpu)

        import cpuinfo
        cpu = cpuinfo.get_cpu_info()
        self.set_system_info('cpu_name', cpu['brand'])
        self.set_system_info('cpu', [cpu['hz_actual_raw'][0], cpu['count']])
