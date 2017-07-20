from __future__ import absolute_import
from __future__ import print_function

import subprocess
import atexit
import os
import pprint
import socket
import select
from threading import Thread, Lock

import coloredlogs
import logging
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
import zlib

import yaml

from aetros.const import JOB_STATUS
from aetros.git import Git
from aetros.logger import GeneralLogger
from aetros.utils import git, invalid_json_values, read_home_config, read_config

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


class ApiClient():
    def __init__(self, api_host, api_key):
        self.host = api_host
        self.api_key = api_key

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

    def get_url(self, affix):

        url = 'http://%s/api/%s' % (self.host, affix)

        if self.api_key:
            if '?' in url:
                url += '&token=' + self.api_key
            else:
                url += '?token=' + self.api_key

        return url


class BackendClient:
    def __init__(self, host, event_listener, logger):
        """
        :type api_host: string
        :type api_key: string
        :type event_listener: EventListener
        :type job_id: integer
        """
        self.host = host or os.getenv('API_HOST') or 'trainer.aetros.com'
        self.ssh_stream = None

        self.event_listener = event_listener
        self.logger = logger
        self.message_id = 0

        self.api_key = None
        self.job_id = None

        self.thread_read_instance = None
        self.thread_write_instance = None

        self.lock = Lock()
        self.connection_errors = 0
        self.queue = []
        self.initial_connection_tries = 0

        self.active = False
        self.external_stopped = False
        self.registered = False
        self.connected = False
        self.read_unpacker = msgpack.Unpacker(encoding='utf8')

    def start(self):
        self.active = True

        if not self.thread_read_instance:
            self.thread_read_instance = Thread(target=self.thread_read)
            self.thread_read_instance.daemon = True
            self.thread_read_instance.start()

        if not self.thread_write_instance:
            self.thread_write_instance = Thread(target=self.thread_write)
            self.thread_write_instance.daemon = True
            self.thread_write_instance.start()

    def on_connect(self):
        pass

    def connect(self):
        if self.initial_connection_tries > 3:
            # We only try in 5 minutes again
            return False

        self.lock.acquire()
        self.initial_connection_tries += 1

        if self.connected:
            self.lock.release()
            return True

        try:
            self.logger.info("Connecting to " + self.host)

            def preexec_function():
                # Ignore the SIGINT signal by setting the handler to the standard
                # signal handler SIG_IGN.
                signal.signal(signal.SIGINT, signal.SIG_IGN)

            self.ssh_stream = subprocess.Popen(['ssh', 'git@' + self.host, 'stream'],
                                               stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                                               preexec_fn=preexec_function)
            self.connected = True
            self.lock.release()

            return self.on_connect()
        except Exception as error:

            if hasattr(error, 'message'):
                self.logger.error("Connection error during connecting to %s: %s" % (self.host, error.message))
            else:
                self.logger.error("Connection error during connecting to %s." % (self.host,))
            time.sleep(1)
            return False
        finally:
            if self.lock.locked():
                self.lock.release()

    def debug(self):
        sent = len(filter(lambda x: x['_sent'], self.queue))
        sending = len(filter(lambda x: x['_sending'], self.queue))
        open = len(filter(lambda x: not x['_sending'], self.queue))
        self.logger.debug("%d sent, %d in sending, %d open " % (sent, sending, open))

    def connection_error(self, error=None):
        if not self.connected:
            return

        if socket is None:
            # python interpreter is already dying, so quit
            return

        message = "Connection error to " + self.host + ": "

        if hasattr(error, 'errno') and hasattr(error, 'message'):
            print(message + "%d: %s" % (error.errno, error.message,))
        elif error:
            print(message + str(error))
        else:
            print(message)

        self.connected = False
        self.registered = False

        self.event_listener.fire('disconnect')

        # set all messages that are in sending to sending=false
        for message in self.queue:
            if message['_sending'] and not message['_sent']:
                message['_sending'] = False

        self.connection_errors += 1

    def thread_write(self):

        while self.active:
            if self.connected and self.registered:
                self.lock.acquire()
                try:
                    sent_size = 0
                    for message in self.queue:
                        if not message['_sending'] and not message['_sent']:
                            size = self.send_message(message)
                            if size is not False:
                                sent_size += size
                                if sent_size > 1024 * 1024:
                                    # not too much at once (max 1MB), so we have time to listen for incoming messages
                                    break

                            self.queue.remove(message)

                finally:
                    self.lock.release()

            time.sleep(0.1)

    def thread_read(self):
        while self.active:
            if self.connected and self.registered:
                try:
                    messages = self.read() # this blocks, so no lock before

                    self.lock.acquire()
                    if messages is not None:
                        self.handle_messages(messages)
                finally:
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

        self.ssh_stream.kill()

    def send(self, message):
        self.message_id += 1
        message['_id'] = self.message_id
        message['_sending'] = False
        message['_sent'] = False
        self.queue.append(message)

    def send_message(self, message):
        if isinstance(message, dict):
            message['_sending'] = True

        import msgpack
        msg = msgpack.packb(message, default=invalid_json_values)

        try:
            self.ssh_stream.stdin.write(msg)
            self.ssh_stream.stdin.flush()
            message['sent'] = True

            return len(msg)
        except Exception as error:
            self.connection_error(error)
            return False

    def handle_messages(self, messages):
        for message in messages:
            if not self.external_stopped and 'STOP' == message:
                self.external_stopped = True
                self.event_listener.fire('stop')

    def wait_for_at_least_one_message(self):
        """
        Reads until we receive at least one message we can unpack. Return all found messages.
        """

        unpacker = msgpack.Unpacker(encoding='utf8')

        while True:
            chunk = ''
            try:
                self.lock.acquire()
                chunk = self.ssh_stream.stdout.read(1)
            except Exception as error:
                self.connection_error(error)
                raise error
            finally:
                self.lock.release()

            if chunk == '':
                # happens only when connection broke. If nothing is to be received, it hangs instead.
                self.connection_error('Connection broken')
                return False

            unpacker.feed(chunk)

            messages = [m for m in unpacker]
            if messages:
                return messages

    def read(self):
        """
        Reads from the socket and tries to unpack the message. If successful (because msgpack was able to unpack)
        then we return that message. Else None. Keep calling .read() when new data is available so we try it 
        again.
        """

        try:
            chunk = self.ssh_stream.stdout.read(1)
        except Exception as error:
            self.connection_error(error)
            raise error

        if chunk == '':
            # socket connection broken
            self.connection_error('Connection broken')
            return None

        # self.read_buffer.seek(0, 2) #make sure we write at the end
        self.read_unpacker.feed(chunk)

        # self.read_buffer.seek(0)
        messages = [m for m in self.read_unpacker]

        return messages if messages else None


class JobClient(BackendClient):
    def configure(self, model_name, job_id):
        self.model_name = model_name
        self.job_id = job_id

    def on_connect(self):
        self.send_message({'type': 'register_job_worker', 'model': self.model_name, 'job': self.job_id})
        self.logger.info("Wait for one message")
        messages = self.wait_for_at_least_one_message()
        self.logger.info("Got: " + str(messages))

        if not messages:
            return False

        message = messages.pop(0)
        self.logger.info("handle message: " + str(message))
        if isinstance(message, dict) and 'a' in message:
            if 'aborted' == message['a']:
                self.logger.error("Job aborted meanwhile. Exiting")
                self.event_listener.fire('stop')
                self.active = False
                return False

            if 'registration_failed' == message['a']:
                self.event_listener.fire('registration_failed', {'reason': message['reason']})
                return False

            if 'registered' == message['a']:
                self.registered = True
                self.logger.info("Connected to %s " % (self.host,))
                self.event_listener.fire('registration')
                self.handle_messages(messages)
                return True

        self.logger.error("Registration of job %s failed." % (self.job_id,))
        return False


def start_job(name=None):
    """
    Tries to load the job defined in the AETROS_JOB_ID environment variable.
    If not defined, it creates a new job.
    Starts the job as well.

    :param name: string: model name
    :return: JobBackend
    """

    job = JobBackend(name)
    job.setup_std_output_logging()

    if os.getenv('AETROS_JOB_ID'):
        job.load(os.getenv('AETROS_JOB_ID'))
    else:
        job.create()

    job.start()

    return job


def create_job(name):
    """
    Creates a new job.

    :param name: string : model name
    :return: JobBackend
    """
    job = JobBackend(name)
    job.create()

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

        self.job_backend.git.commit_json_file('CREATE_CHANNEL', 'aetros/job/channel/' + name+ '/config', message)
        self.stream = self.job_backend.git.stream_file('aetros/job/channel/' + name+ '/data')
        self.stream.write("x;training;validation\n")

    def send(self, x, training_loss, validation_loss):
        line = ";".join([str(x), str(training_loss), str(validation_loss)])
        self.stream.write(line + "\n")


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
        self.job_backend.git.commit_json_file('CREATE_CHANNEL', 'aetros/job/channel/' + name+ '/config', message)
        self.stream = self.job_backend.git.stream_file('aetros/job/channel/' + name+ '/data')

        line = "x;".join(str(x['name']) for x in traces)
        self.stream.write(line + "\n")

    def send(self, x, y):
        if not isinstance(y, list):
            y = [y]

        if len(y) != len(self.traces):
            raise Exception(
                'You tried to set more y values (%d items) then traces available in channel %s (%d traces).' % (
                    len(y), self.name, len(self.traces)))

        line = str(x) + ";".join(str(x) for x in y)
        self.stream.write(line + "\n")


class JobBackend:
    """
    :type event_listener: EventListener
    :type id: str: model name
    :type job_id: str: job id
    :type client: Client
    :type job: dict
    """

    def __init__(self, model_name=None):
        self.event_listener = EventListener()

        on_shutdown.started_jobs.append(self)

        self.log_file_handle = None
        self.config = {}
        self.job = {'parameters': {}}

        self.git = None

        self.ssh_stream = None
        self.model_name = model_name

        self.client = None
        self.stream_log = None

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

        self.logger = logging.getLogger('aetros')
        coloredlogs.install(level='INFO', logger=self.logger)

        self.general_logger_stdout = GeneralLogger(job_backend=self)
        self.general_logger_error = GeneralLogger(job_backend=self, error=True)

        self.last_progress_call = None
        self.job_ids = []
        self.in_request = False
        self.stop_requested = False

        self.event_listener.on('stop', self.external_stop)
        self.event_listener.on('aborted', self.external_aborted)
        self.event_listener.on('registration', self.on_registration)
        self.event_listener.on('registration_failed', self.on_registration_failed)

        self.read_config()
        self.client = JobClient(self.host, self.event_listener, self.logger)
        self.git = Git(self.logger, self.client, self.host, self.git_path, self.model_name)


    @property
    def git_path(self):
        return self.config['git_path']

    @property
    def host(self):
        return self.config['host']

    def on_registration_failed(self, params):
        self.logger.warning("Connecting to AETROS Trainer at %s failed. Reasons: %s. Wrong ssh key?" % (self.host, params['reason'],))
        self.logger.warning("Live streaming disabled. Job continues to local git only. Use 'aetros push-job %s' to public your job when done." % (self.job_id, ), file=sys.stderr)

    def on_registration(self, params):
        self.logger.info("Job %s#%s started. Open http://%s/job/%s to monitor the training." %
              (self.model_name, self.job_id, self.host, self.job_id))

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
        self.logger.warning("Job stopped through AETROS Trainer.")
        self.abort()
        self.stop_requested = True
        os.kill(os.getpid(), signal.SIGINT)

    def batch(self, batch, total, size=None):
        time_diff = time.time() - self.last_batch_time
        self.made_batches += 1

        if time_diff > 1 or batch == total:  # only each second second or last batch
            with self.git.batch_commit('BATCH'):
                self.set_system_info('currentBatch', batch, True)
                self.set_system_info('batchSize', size, True)
                self.set_system_info('nb_batches', total, True)

                self.batches_per_second = self.made_batches / time_diff
                self.made_batches = 0
                self.last_batch_time = time.time()

                if size:
                    self.set_system_info('samplesPerSecond', self.batches_per_second * size, True)

                epochs_per_second = self.batches_per_second / total  # all batches
                self.set_system_info('epochsPerSecond', epochs_per_second, True)

                elapsed = time.time() - self.start_time
                self.set_system_info('elapsed', elapsed, True)

                if self.total_epochs:
                    eta = 0
                    if batch < total:
                        # time to end this epoch
                        eta += (total - batch) / self.batches_per_second

                    # time until all epochs are done
                    if self.total_epochs - (self.current_epoch - 1) > 0:
                        eta += (self.total_epochs - (self.current_epoch - 1)) / epochs_per_second

                    self.set_system_info('eta', eta, True)

        self.current_batch = batch

    @property
    def job_settings(self):
        if 'settings' in self.job['config']:
            return self.job['config']['settings']
        return {}

    def progress(self, epoch, total):
        self.current_epoch = epoch
        self.total_epochs = total
        epoch_limit = False

        with self.git.batch_commit('PROGRESS ' + str(epoch)):
            if 'maxEpochs' in self.job_settings and self.job_settings['maxEpochs'] > 0:
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
                    self.logger.info("\nMaxEpochs of %d/%d reached" % (epoch, self.total_epochs))
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

        self.logger.info('Job backend start')
        self.client.configure(self.model_name, self.job_id)
        self.client.start()

        with self.git.batch_commit('JOB_STARTED'):
            self.git.commit_json_file('JOB_STARTED', 'aetros/job/started', time.time())
            self.job_add_status('progress', JOB_STATUS.PROGRESS_STATUS_STARTED)

        self.stream_log = self.git.stream_file('aetros/job/log')

        self.running = True
        self.ended = False
        self.collect_system_information()
        self.collect_environment()
        #todo self.start_monitoring()


        self.detect_git_version()

    def detect_git_version(self):
        try:
            commit_sha = git.get_current_commit_hash()
            if commit_sha:
                self.set_system_info('git_version', commit_sha)

            current_branch = git.get_current_branch()
            if current_branch:
                self.set_system_info('git_branch', current_branch)
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

        self.job_add_status('progress', JOB_STATUS.PROGRESS_STATUS_DONE)

        if self.total_epochs:
            self.progress(self.total_epochs, self.total_epochs)

        self.stop(True)

    def stop(self, wait_for_client=False):
        if self.monitoring_thread:
            self.monitoring_thread.stop()

        self.logger.info("stopping ...")
        self.ended = True
        self.running = False

        self.logger.info("git stopping ...")
        self.git.stop()

        self.logger.info("client stopping ...")
        if wait_for_client:
            self.client.end()
        else:
            self.client.close()
        self.logger.info("Stopped.")

    def abort(self):
        if not self.running:
            return

        with self.git.batch_commit('ABORTED'):
            self.job_add_status('progress', JOB_STATUS.PROGRESS_STATUS_ABORTED)
            self.set_status('ABORTED')

        self.stop()

    def crash(self, error=None):
        with self.git.batch_commit('CRASH'):
            self.job_add_status('progress', JOB_STATUS.PROGRESS_STATUS_CRASHED)
            self.set_status('CRASHED')
            self.git.commit_json_file('CRASH_REPORT_ERROR', 'aetros/job/crash/error', str(error) if error else '')
            self.git.commit_json_file('CRASH_REPORT_LAST_MESSAGE', 'aetros/job/crash/last_message', self.general_logger_error.last_messages)

        self.stop()

    def write_log(self, message):
        if self.stream_log:
            self.stream_log.write(message)

    def set_status(self, status):
        self.logger.info('Job status changed to %s ' % (status,))
        self.job_add_status('status', status)

    @property
    def git_host(self):
        return 'dev.aetros.com'

    @property
    def ssh_url(self):
        return 'git@%s' % (self.git_host,)

    @property
    def job_id(self):
        return self.git.job_id

    def create(self, hyperparameter=None, server='local', dataset_id=None, insights=False):
        self.git.create_job_id()

        self.job['id'] = self.job_id
        self.job['name'] = self.model_name
        self.job['datasetId'] = dataset_id
        self.job['insights'] = insights
        self.job['server'] = server
        self.job['config'] = self.config
        self.job['optimization'] = None

        if hyperparameter:
            self.job['parameters'] = hyperparameter

        # get model config from where?
        with self.git.batch_commit('JOB_CONFIG'):
            self.git.commit_json_file('JOB', 'aetros/job.json', self.job)
            self.job_add_status('progress', JOB_STATUS.PROGRESS_STATUS_CREATED)

        return self.job_id

    def is_keras_model(self):
        if not self.job:
            raise Exception('Job not loaded yet. Use load(id) first.')

        return not self.job['config']['fromCode']

    def read_config(self):
        self.config = read_config(logger=self.logger)

        if 'parameters' in self.config:
            self.job['parameters'] = self.config['parameters']

        if not self.model_name and ('model' in self.job or not self.job['model']):
            raise Exception('No model name given. Specify in .aetros.yml or in aetros.backend.start_job("model/name")')

    def get_parameter(self, name, default=None):
        params = self.job['parameters']

        if name in params:
            return params[name]

        if '.' in name:
            path = name.split('.')
            current = params
            for item in path:
                if item not in current:
                    if default is None:
                        raise Exception('Parameter ' + name + ' not found and no default value given.')

                    return default

                current = current[item]

            return current

        if default is None:
            raise Exception('Parameter ' + name + ' not found and no default value given.')

        return default

    def restart(self, job_id):
        """
        Restarts job.
        :param id: int
        """
        self.git.job_id = job_id

        if not self.job_id:
            raise Exception('No job id given.')

        # response = self.post('job/restart', {'id': self.job_id})

        # if response.status_code != 200:
        #     raise Exception("Could not restart job: %s" % (response.content,))

    def load(self, job_id):
        """
        Loads job and sets as current.
        :param job_id: int
        """

        self.git.job_id = job_id

        self.job = self.git.git_read('aetros/job.json')
        print('self.job', self.job)

        # todo, patch self.job['config'] from already existing
        # todo, patch self.job['parameters'] from already existing
        # this is necessary, when the Trainer created the job already and should now be started.

        raise Exception('no implemented')


        # if not self.job_id:
        #     raise Exception('No job id given.')
        #
        # response = self.get('job', {'id': self.job_id})
        #
        # if response.status_code != 200:
        #     raise Exception("Could not find job: %s" % (response.content,))
        #
        # job = response.json()
        #
        # if job is None or 'error' in job:
        #     raise Exception('Job not found. Have you configured your token correctly? %s: %s' %
        #                     (job['error'], job['message']))
        #
        # self.job = response.json()
        # self.job_id = self.job['id']

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

        raise Exception('not implemeneted')
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
        self.git.commit_json_file('STATUS ' + key, 'aetros/job/status/' + key, value)

    def set_info(self, key, value):
        self.git.commit_json_file('INFO ' + key, 'aetros/job/info/' + key, value)

    def set_graph(self, graph):
        self.client.send({
            'type': 'job-graph',
            'data': {'graph': graph}
        })

    def set_system_info(self, key, value, commit_end_of_job=False):
        if commit_end_of_job:
            self.git.store_file('aetros/job/system/' + key, json.dumps(value, default=invalid_json_values))
        else:
            self.git.commit_json_file('SYSTEM_INFO ' + key, 'aetros/job/system/' + key, value)

    def commit_file(self, file_path, name=None):
        if os.path.getsize(file_path) > 10 * 1024 * 1024 * 1024:
            raise Exception('Can not upload file bigger than 10MB')

        with open(file_path, 'rb') as f:
            contents = f.read()

        self.git.commit_file('FILE ' + (name or file_path), os.path.expanduser(file_path), contents)

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

        with self.git.batch_commit('JOB_SYSTEM_INFORMATION'):
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
