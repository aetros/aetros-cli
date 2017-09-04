from __future__ import absolute_import
from __future__ import print_function

import subprocess
import atexit
import os
import socket
from threading import Thread, Lock

import coloredlogs
import logging

import copy
import requests
import signal
import json
import time
import six
import PIL.Image
import sys
import msgpack

from aetros.const import JOB_STATUS
from aetros.git import Git
from aetros.logger import GeneralLogger
from aetros.utils import git, invalid_json_values, read_config

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


class ApiClient:
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
    def __init__(self, config, event_listener, logger):
        """
        :type api_host: string
        :type api_key: string
        :type event_listener: EventListener
        :type job_id: integer
        """
        self.host = config['host']
        self.ssh_stream = None
        self.ssh_command = config['ssh']
        self.ssh_key_path = config['ssh_key']

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
        self.connection_tries = 0
        self.in_connecting = False

        # indicates whether we are offline or not, means not connection to the internet and
        # should not establish a connection to Aetros.
        self.online = True

        # Whether the client is active and should do things.
        self.active = False
        self.expect_close = False
        self.external_stopped = False
        self.registered = False
        self.connected = False
        self.was_connected_once = False
        self.read_unpacker = msgpack.Unpacker(encoding='utf-8')

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

    def on_connect(self, reconnect=False):
        pass

    def go_offline(self):
        if not self.online:
            return

        self.event_listener.fire('offline')
        self.online = False

    def connect(self):
        """
        In the write-thread we detect that no connection is living anymore and try always again.
        Up to the 3 connection try, we report to user. We keep trying but in silence.
        Also, when more than 10 connection tries are detected, we delay extra 60 seconds.
        """
        if self.connection_tries > 10:
            # We only try in 1 minute again
            time.sleep(60)

        if self.in_connecting:
            return False

        self.in_connecting = True

        self.logger.debug('Wanna connect ...')

        try:
            if self.connected or not self.online:
                return True

            def preexec_fn():
                # Ignore the SIGINT signal by setting the handler to the standard
                # signal handler SIG_IGN.
                signal.signal(signal.SIGINT, signal.SIG_IGN)

            args = [self.ssh_command] if isinstance(self.ssh_command, six.string_types) else self.ssh_command
            args += ['-o', 'StrictHostKeyChecking no']

            if self.ssh_key_path:
                args += ['-i', self.ssh_key_path]

            args = args + ['git@' + self.host, 'stream']
            self.logger.debug('Open ssh: ' + (' '.join(args)))

            self.ssh_stream = subprocess.Popen(args, preexec_fn=preexec_fn, bufsize=0,
                                               stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            self.logger.debug('Done: pid=' + str(self.ssh_stream.pid))
            messages = self.wait_for_at_least_one_message()
            stderrdata = ''

            if not messages:
                stderrdata = self.ssh_stream.stderr.read().decode("utf-8").strip()
            else:
                self.connected = True
                self.registered = self.on_connect(self.was_connected_once)

            if not self.registered:
                self.connected = False
                self.ssh_stream.kill()
                self.connection_tries += 1
                self.go_offline()

                if stderrdata:
                    if 'Connection refused' not in stderrdata and 'Permission denied' not in stderrdata:
                        self.logger.error(stderrdata)

                if 'Permission denied' in stderrdata:
                    if self.connection_tries < 3:
                        self.logger.warning("Access denied. Did you setup your ssh key correctly and saved it in your AETROS Trainer user account?")

                    self.close()
                    sys.exit(1)
            else:
                self.was_connected_once = True

        except Exception as error:
            self.connected = False
            self.registered = False
            if self.connection_tries < 3:
                self.logger.error("Connection error during connecting to %s: %s" % (self.host, str(error)))
            raise
        finally:
            self.in_connecting = False

        return self.connected

    def debug(self):
        sent = len(filter(lambda x: x['_sent'], self.queue))
        sending = len(filter(lambda x: x['_sending'], self.queue))
        open = len(filter(lambda x: not x['_sending'], self.queue))
        self.logger.debug("%d sent, %d in sending, %d open " % (sent, sending, open))

    def end(self):
        self.expect_close = True
        self.send_message({'type': 'end'})
        self.wait_for_close()

    def connection_error(self, error=None):
        if not self.active:
            return

        if self.expect_close:
            return

        if not self.connected:
            return

        # needs to be set before logger.error, since they can call send_message again
        self.connected = False
        self.registered = False

        if socket is None:
            # python interpreter is already dying, so quit
            return

        message = "Connection error"

        if error:
            self.logger.error(message + ": " + str(error))
        else:
            self.logger.error(message)

        import traceback
        self.logger.debug(traceback.format_exc())

        self.event_listener.fire('disconnect')
        self.connection_errors += 1

    def thread_write(self):
        while self.active:
            if self.online:
                if self.connected and self.registered:
                    queue_copy = self.queue[:]

                    try:
                        sent_size = 0
                        sent = []

                        for message in queue_copy:
                            if message['_sending'] and not message['_sent']:
                                message['_sending'] = False

                        for message in queue_copy:
                            if not self.connected or not self.registered:
                                # additional check to make sure there's no race condition
                                break

                            if not message['_sending'] and not message['_sent']:
                                size = self.send_message(message)
                                if size is not False:
                                    sent.append(message)

                                    sent_size += size
                                    if sent_size > 1024 * 1024:
                                        # not too much at once (max 1MB), so we have time to listen for incoming messages
                                        break

                        self.lock.acquire()
                        for message in sent:
                            if message in self.queue:
                                self.queue.remove(message)
                        self.lock.release()
                    except SystemExit:
                        return
                    except KeyboardInterrupt:
                        return
                    finally:
                        if self.lock.locked():
                            self.lock.release()

                if not self.connected and not self.expect_close:
                    if not self.connect():
                        time.sleep(5)

            time.sleep(0.01)

    def thread_read(self):
        while self.active:
            if self.online:
                if self.connected and self.registered:
                    try:
                        messages = self.read() # this blocks

                        if messages is not None:
                            self.handle_messages(messages)

                        continue
                    except SystemExit:
                        return
                    except KeyboardInterrupt:
                        return
                    except:
                        pass

            time.sleep(0.01)

    def wait_sending_last_messages(self):
        # send all missing messages
        i = 0
        while self.registered:
            if len(self.queue) == 0:
                break

            i += 1
            time.sleep(0.1)
            if i % 50 == 0:
                self.logger.warning("We still sync job data. %d messages left. " % (len(self.queue),))

    def wait_for_close(self):
        self.active = False

        if not self.online:
            return

        if not self.ssh_stream:
            return

        while self.ssh_stream.poll() is None:
            time.sleep(0.1)

        self.online = False

    def close(self):
        self.active = False
        self.connected = False

        self.event_listener.fire('close')

        if self.ssh_stream:
            self.ssh_stream.kill()

        self.online = False

    def send(self, message):
        if not self.active:
            return
            # raise Exception("Requested to send messages, but client is inactive.")

        self.message_id += 1
        message['_id'] = self.message_id
        message['_sending'] = False
        message['_sent'] = False

        self.queue.append(message)

    def send_message(self, message):
        if not self.connected:
            return False

        message['_sending'] = True

        import msgpack
        msg = msgpack.packb(message, default=invalid_json_values)

        try:
            self.ssh_stream.stdin.write(msg)
            message['sent'] = True
            self.ssh_stream.stdin.flush()

            return len(msg)
        except KeyboardInterrupt:
            if message['sent']:
                return len(msg)

            return False

        except Exception as error:
            self.connection_error(error)
            return False

    def handle_messages(self, messages):
        for message in messages:
            if not self.external_stopped and 'stop' == message['a']:
                self.external_stopped = True
                self.event_listener.fire('stop', message['force'])

    def wait_for_at_least_one_message(self):
        """
        Reads until we receive at least one message we can unpack. Return all found messages.
        """

        unpacker = msgpack.Unpacker(encoding='utf-8')

        while True:
            chunk = ''
            try:
                chunk = self.ssh_stream.stdout.read(1)
            except Exception as error:
                self.connection_error(error)
                raise

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
            raise

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

    def on_connect(self, reconnect=False):
        self.send_message({'type': 'register_job_worker', 'model': self.model_name, 'job': self.job_id, 'reconnect': reconnect})
        self.logger.debug("Wait for one client registration")
        messages = self.wait_for_at_least_one_message()
        self.logger.debug("Got " + str(messages))

        if not messages:
            self.event_listener.fire('registration_failed', {'reason': 'No answer received.'})
            return False

        message = messages.pop(0)
        self.logger.debug("handle message: " + str(message))
        if isinstance(message, dict) and 'a' in message:
            if 'aborted' == message['a']:
                self.logger.error("Job aborted meanwhile. Exiting")
                self.event_listener.fire('stop', False)
                self.active = False
                return False

            if 'registration_failed' == message['a']:
                self.event_listener.fire('registration_failed', {'reason': message['reason']})
                return False

            if 'registered' == message['a']:
                self.registered = True
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
            'traces': [{'name': 'training'}, {'name': 'validation'}],
            'type': JobChannel.NUMBER,
            'main': True,
            'xaxis': xaxis,
            'yaxis': yaxis,
            'layout': layout,
            'lossChannel': True
        }

        self.job_backend.git.commit_json_file('CREATE_CHANNEL', 'aetros/job/channel/' + name+ '/config', message)
        self.stream = self.job_backend.git.stream_file('aetros/job/channel/' + name+ '/data.csv')
        self.stream.write('"x","training","validation"\n')

    def send(self, x, training_loss, validation_loss):
        line = json.dumps([x, training_loss, validation_loss])[1:-1]
        self.stream.write(line + "\n")
        self.job_backend.git.store_file('aetros/job/channel/' + self.name + '/last.csv', line)


class JobImage:
    def __init__(self, id, image, label=None):
        self.id = id
        if not isinstance(image, PIL.Image.Image):
            raise Exception('JobImage requires a PIL.Image as image argument.')

        self.image = image
        self.label = label


class JobChannel:
    NUMBER = 'number'
    TEXT = 'text'

    """
    :type job_backend: JobBackend
    """

    def __init__(self, job_backend, name, traces=None,
                 main=False, kpi=False, kpiTrace=0, max_optimization=True,
                 type=None, xaxis=None, yaxis=None, layout=None):
        """
        :param job_backend: JobBakend
        :param name: str
        :param traces: None|list : per default create a trace based on "name".
        :param main: bool : whether this channel is visible in the job list as column for better comparison.

        :param kpi: bool : whether this channel is the KPI (key performance indicator).
                           Used for hyperparameter optimization. Only one channel can be a kpi. Only first trace used.
        :param kpiTrace: bool : if you have multiple traces, define which is the KPI. 0 based index.

        :param max_optimization: bool : whether the optimization maximizes or minmizes the kpi . Use max_optimization=False to
                                        tell the optimization algorithm that his channel minimizes a kpi, for instance the loss of a model.

        :param type: str : One of JobChannel.NUMBER, JobChannel.TEXT, JobChannel.IMAGE
        :param xaxis: dict
        :param yaxis: dict
        :param layout: dict
        """
        self.name = name
        self.job_backend = job_backend
        self.kpi = kpi
        self.kpiTrace = kpiTrace

        if self.kpi:
            self.job_backend.kpi_channel = self

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
            'kpiTrace': kpiTrace,
            'maxOptimization': max_optimization,
            'xaxis': xaxis,
            'yaxis': yaxis,
            'layout': layout,
        }
        self.traces = traces
        self.job_backend.git.commit_json_file('CREATE_CHANNEL', 'aetros/job/channel/' + name+ '/config', message)
        self.stream = self.job_backend.git.stream_file('aetros/job/channel/' + name+ '/data.csv')

        if self.kpi:
            self.job_backend.git.commit_file('KPI_CHANNEL', 'aetros/job/kpi/name', name)

        line = json.dumps(['x'] + [str(x['name']) for x in traces])[1:-1]
        self.stream.write(line + "\n")

    def send(self, x, y):
        if not isinstance(y, list):
            y = [y]

        if len(y) != len(self.traces):
            raise Exception(
                'You tried to set more y values (%d items) then traces available in channel %s (%d traces).' % (
                    len(y), self.name, len(self.traces)))

        line = json.dumps([x] + y)[1:-1]
        self.stream.write(line + "\n")
        self.job_backend.git.store_file('aetros/job/channel/' + self.name + '/last.csv', line)

        if self.kpi:
            self.job_backend.git.store_file('aetros/job/kpi/last.json', json.dumps(y[self.kpiTrace]))


class JobBackend:
    """
    :type event_listener: EventListener
    :type id: str: model name
    :type job_id: str: job id
    :type client: Client
    :type job: dict
    """

    def __init__(self, model_name=None, logger=None):
        self.event_listener = EventListener()

        self.log_file_handle = None
        self.config = {}
        self.job = {'parameters': {}}

        self.git = None

        self.ssh_stream = None
        self.model_name = model_name
        self.logger = logger

        self.client = None
        self.stream_log = None

        self.last_batch_time = time.time()
        self.start_time = time.time()
        self.current_epoch = 0
        self.current_batch = 0
        self.total_epochs = 0
        self.made_batches = 0
        self.batches_per_second = 0

        # indicates whether early_stop() has been called. Is called by reaching maxTimes or maxEpochs limitation.
        # This flag stops exiting with > 0, since the reach of a limitation is a valid exit.
        self.in_early_stop = False

        # done means: done, abort or crash method has been called.
        self.ended = False

        # when stop(wait_for_client=True) is called, we sync last messages.
        # this flag indicates that we are not yet ended completely but in the progress
        self.stopping = False
        # running means: the syncer client is running.
        self.running = False
        self.monitoring_thread = None

        if not self.logger:
            self.logger = logging.getLogger('aetros-job')
            coloredlogs.install(level=self.log_level, logger=self.logger)

        self.general_logger_stdout = GeneralLogger(job_backend=self)
        self.general_logger_error = GeneralLogger(job_backend=self, error=True)

        self.last_progress_call = None
        self.job_ids = []
        self.in_request = False
        self.stop_requested = False
        self.stop_requested_force = False

        self.kpi_channel = None
        self.registered = None
        self.progresses = {}

        self.event_listener.on('stop', self.external_stop)
        self.event_listener.on('aborted', self.external_aborted)
        self.event_listener.on('registration', self.on_registration)
        self.event_listener.on('registration_failed', self.on_registration_failed)
        self.event_listener.on('offline', self.on_client_offline)

        signal.signal(signal.SIGINT, self.on_signint)
        signal.signal(signal.SIGTERM, self.on_signint)
        signal.signal(signal.SIGUSR1, self.on_signusr1)

        self.pid = os.getpid()

        self.read_config(model_name)
        self.client = JobClient(self.config, self.event_listener, self.logger)
        self.git = Git(self.logger, self.client, self.host, self.config['storage_dir'], self.model_name)

    @property
    def log_level(self):
        if os.getenv('DEBUG') == '1':
            return 'DEBUG'

        return 'INFO'

    @property
    def host(self):
        return self.config['host']

    def on_registration_failed(self, params):
        self.logger.warning("Connecting to AETROS Trainer at %s failed. Reasons: %s" % (self.host, params['reason'],))
        if 'Permission denied' in params['reason']:
            self.logger.warning("Make sure you have saved your ssh pub key in your AETROS Trainer user account.")

    def on_client_offline(self, params):
        self.logger.warning("Could not establish a connection. We stopped automatic syncing.")
        self.logger.warning("You can publish later this job to AETROS Trainer using following command in this folder.")
        self.logger.warning("$ aetros push-job " + self.model_name + "/" + self.job_id)
        self.git.online = False

    def on_registration(self, params):
        if not self.registered:
            self.registered = True
            self.logger.info("Job %s/%s started." % (self.model_name, self.job_id))
            self.logger.info("Open http://%s/model/%s/job/%s to monitor the training." % (self.host, self.model_name, self.job_id))

            self.logger.debug('Git backend start')
            self.git.start()
        else:
            self.logger.info("Successfully reconnected.")

    def on_signusr1(self, signal, frame):
        self.logger.warning("USR1: backend (running=%s, ended=%s), client (online=%s, active=%s, registered=%s, "
                            "connected=%s, queue=%d), git (online=%s, active_thread=%s, last_push_time=%s)." % (
          str(self.running),
          str(self.ended),
          str(self.client.online),
          str(self.client.active),
          str(self.client.registered),
          str(self.client.connected),
          len(self.client.queue),
          str(self.git.online),
          str(self.git.active_thread),
          str(self.git.last_push_time),
        ))

    def on_signint(self, sig, frame):
        if self.stop_requested:
            self.stop_requested_force = True
            self.logger.warning('Force stopped')
            sys.exit(1)

        self.stop_requested = True

        self.logger.warning('Received SIGINT signal. Press CTRL+C again to force stop. Stopping ...')
        self.abort(True)
        sys.exit(1)

    def external_aborted(self, params):
        """
        Immediately abort the job
        """
        self.ended = True
        self.running = False

        if self.monitoring_thread:
            self.monitoring_thread.stop()

        self.git.stop()
        self.client.close()

        sys.exit(1)

    def external_stop(self, params):
        """
        Stop through GUI
        """
        if self.stop_requested:
            return

        self.logger.warning("Sent SIGINT through AETROS Trainer.")
        self.stop_requested_force = params
        os.kill(os.getpid(), signal.SIGINT)

    def setup_std_output_logging(self):
        """
        Overwrites sys.stdout and sys.stderr so we can send it additionally to Aetros.
        """
        sys.stdout = self.general_logger_stdout
        sys.stderr = self.general_logger_error

    def batch(self, batch, total, size=None):
        time_diff = time.time() - self.last_batch_time
        self.made_batches += 1

        if time_diff > 1 or batch == total:  # only each second second or last batch
            with self.git.batch_commit('BATCH'):
                self.set_system_info('batch', batch, True)
                self.set_system_info('batches', total, True)
                self.set_system_info('batchSize', size, True)

                self.batches_per_second = self.made_batches / time_diff
                self.made_batches = 0
                self.last_batch_time = time.time()

                if size:
                    self.set_system_info('samplesPerSecond', self.batches_per_second * size, True)

                epochs_per_second = self.batches_per_second / total  # all batches
                self.set_system_info('epochsPerSecond', epochs_per_second, True)

                elapsed = time.time() - self.start_time
                self.set_system_info('elapsed', elapsed, True)

                # self.git.store_file('aetros/job/times/elapsed.json', json.dumps(elapsed))

                if self.total_epochs:
                    eta = 0
                    if batch < total:
                        # time to end this epoch
                        if self.batches_per_second != 0:
                            eta = (total - batch) / self.batches_per_second

                    # time until all epochs are done
                    if self.total_epochs - (self.current_epoch - 1) > 0:
                        if epochs_per_second != 0:
                            eta += (self.total_epochs - (self.current_epoch - 1)) / epochs_per_second

                    self.git.store_file('aetros/job/times/eta.json', json.dumps(eta))

        self.current_batch = batch

    @property
    def job_settings(self):
        if 'settings' in self.job['config']:
            return self.job['config']['settings']
        return {}

    def create_progress(self, id, total_steps=100):
        if id in self.progresses:
            return self.progresses[id]

        class Controller():
            def __init__(self, git, id, total_steps=100):
                self.started = False
                self.stopped = False
                self.lock = Lock()
                self.id = id
                self.step = 0
                self.steps = total_steps
                self.eta = 0
                self.last_call = 0
                self.git = git
                self._label = id

                self.store()

            def store(self):
                info = {
                    'label': self._label,
                    'started': self.started,
                    'stopped': self.stopped,
                    'step': self.step,
                    'steps': self.steps,
                    'eta': self.eta,
                }
                self.git.store_file('aetros/job/progress/' + self.id + '.json', json.dumps(info))

            def label(self, label):
                self._label = label
                self.store()

            def start(self):
                if self.started is not False:
                    return

                self.started = time.time()
                self.last_call = time.time()
                self.store()

            def stop(self):
                if self.stopped is not False:
                    return

                self.stopped = time.time()
                self.store()

            def advance(self, steps=1):
                if steps <= 0:
                    return

                self.lock.acquire()
                if self.started is False:
                    self.start()

                took = (time.time() - self.last_call) / steps

                self.last_call = time.time()
                self.step += steps
                self.eta = took * (self.steps - self.step)

                if self.step >= self.steps:
                    self.stop()

                self.store()
                self.lock.release()

        self.progresses[id] = Controller(self.git, id, total_steps)

        return self.progresses[id]

    def progress(self, epoch, total):
        self.current_epoch = epoch
        self.total_epochs = total
        epoch_limit = False

        if 'maxEpochs' in self.job['config'] and self.job['config']['maxEpochs'] > 0:
            epoch_limit = True
            self.total_epochs = self.job['config']['maxEpochs']

        if epoch is not 0 and self.last_progress_call:
            # how long took it since the last call?
            time_per_epoch = time.time() - self.last_progress_call
            eta = time_per_epoch * (self.total_epochs - epoch)
            if epoch > total:
                eta = 0

            self.git.store_file('aetros/job/times/eta.json', json.dumps(eta))

            if time_per_epoch > 0:
                self.set_system_info('epochsPerSecond', 1 / time_per_epoch, True)

        self.set_system_info('epoch', epoch, True)
        self.set_system_info('epochs', self.total_epochs, True)
        self.last_progress_call = time.time()

        if epoch_limit and self.total_epochs > 0:
            if epoch >= self.total_epochs:
                self.logger.warning("\nMaxEpochs of %d/%d reached" % (epoch, self.total_epochs))
                self.early_stop()
                return

    def create_loss_channel(self, name, xaxis=None, yaxis=None, layout=None):
        """
        :param name: string
        :return: JobLossGraph
        """

        return JobLossChannel(self, name, xaxis, yaxis, layout)

    def create_channel(self, name, traces=None,
                       main=False, kpi=False, kpiTrace=0, max_optimization=True,
                       type=JobChannel.NUMBER,
                       xaxis=None, yaxis=None, layout=None):
        """
        :param name: str
        :param traces: None|list : per default create a trace based on "name".
        :param main: bool : whether this channel is visible in the job list as column for better comparison.

        :param kpi: bool : whether this channel is the KPI (key performance indicator).
                           Used for hyperparameter optimization. Only one channel can be a kpi. Only first trace used.
        :param kpiTrace: bool : if you have multiple traces, define which is the KPI. 0 based index.

        :param max_optimization: bool : whether the optimization maximizes or minmizes the kpi. Use max_optimization=False to
                                        tell the optimization algorithm that his channel minimizes a kpi, for instance the loss of a model.

        :param type: str : One of JobChannel.NUMBER, JobChannel.TEXT
        :param xaxis: dict
        :param yaxis: dict
        :param layout: dict
        """
        return JobChannel(self, name, traces, main, kpi, kpiTrace, max_optimization, type, xaxis, yaxis, layout)

    def start(self):
        if self.running:
            raise Exception('Job already running.')

        if not self.job_id:
            raise Exception('No job id found. Use create() first.')

        if not self.job:
            raise Exception('Job not loaded')

        on_shutdown.started_jobs.append(self)

        self.client.configure(self.model_name, self.job_id)
        if self.git.online:
            self.logger.debug('Job backend start')
            self.client.start()
        else:
            self.logger.debug('Job backend not started, since being online not detected.')

        self.git.commit_file('JOB', 'aetros/job/times/started.json', str(time.time()))
        self.job_add_status('progress', JOB_STATUS.PROGRESS_STATUS_STARTED)
        self.git.store_file('aetros/job/times/elapsed.json', str(0))

        self.stream_log = self.git.stream_file('aetros/job/log.txt')

        self.running = True
        self.ended = False
        self.collect_system_information()
        self.collect_environment()
        self.start_monitoring()

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

    def early_stop(self):
        """
        Stop when a limitatin is reached (like maxEpoch, maxtime)
        """
        self.in_early_stop = True
        self.done()
        os.kill(self.pid, signal.SIGINT)

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

        if insights and (insights_x is None or insights_x is False):
            raise Exception('Can not build Keras callback with active insights but with invalid `insights_x` as input.')

        if confusion_matrix and (validation_data is None or validation_data is False):
            raise Exception('Can not build Keras callback with active confusion_matrix but with invalid `validation_data` as input.')

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
        if self.stopping:
            return

        if self.stop_requested:
            if self.stop_requested_force:
                self.crash('Force stopped.')
            else:
                self.abort()
            return

        if not self.ended and hasattr(sys, 'last_value'):
            # sys.last_value contains a exception, when there was an uncaught one
            if isinstance(sys.last_value, KeyboardInterrupt):
                self.abort()
            else:
                self.crash(type(sys.last_value).__name__ + ': ' + str(sys.last_value))

        elif self.running:
            self.done()

        if self.in_early_stop:
            sys.exit(0)

    def done(self):
        if not self.running:
            return

        self.stop(JOB_STATUS.PROGRESS_STATUS_DONE, True)

    def stop(self, progress, wait_for_client=False):
        if self.monitoring_thread:
            self.monitoring_thread.stop()

        self.stopping = True
        self.logger.debug("stopping ...")
        self.ended = True
        self.running = False

        info_to_send_job = not wait_for_client
        if not info_to_send_job:
            info_to_send_job = len(self.client.queue) > 0 and not self.client.registered

        if self.git.online:
            if wait_for_client:
                self.logger.debug("client send last %d messages ..." % (len(self.client.queue),))
                self.client.wait_sending_last_messages()

        # store all store_file and stream_file as blob
        self.logger.debug("Git stopping ...")
        self.git.stop()

        if self.client.online:
            # server stores now all store_file and stream_file as blob as well,
            # wait for connection close to make sure server stored their files in server's git
            self.logger.debug("client stopping ...")
            self.client.end()

        # everything stopped. Server has all blobs, so we add the last progress
        # and do last git push.
        self.git.commit_file('STOP', 'aetros/job/status/progress.json', json.dumps(progress, default=invalid_json_values))

        # both sides should have same blobs, so we do now our final push
        self.logger.debug("git last push ...")
        successful_push = self.git.push()

        if not successful_push and info_to_send_job:
            self.logger.warning("Not all job information has been uploaded.")
            self.logger.warning("Please run following command to make sure your job is stored on the server.")
            self.logger.warning("$ aetros push-job " + self.model_name + "/" + self.job_id)

        # remove the index file
        self.git.clean_up()

        self.stopping = False
        self.logger.info("Stopped %s with last commit %s." % (self.git.ref_head, self.git.git_last_commit))

    def abort(self, wait_for_client_messages=True):
        if not self.running:
            return

        self.set_status('ABORTED')

        self.stop(JOB_STATUS.PROGRESS_STATUS_ABORTED, wait_for_client_messages)

    def crash(self, error=None):
        with self.git.batch_commit('CRASH'):
            self.set_status('CRASHED')
            self.git.commit_json_file('CRASH_REPORT_ERROR', 'aetros/job/crash/error', str(error) if error else '')
            self.git.commit_json_file('CRASH_REPORT_LAST_MESSAGE', 'aetros/job/crash/last_message', self.general_logger_error.last_messages)

        # we need to make sure that the server got the git crash before we report the crash progress
        self.git.push()

        self.job_add_status('progress', JOB_STATUS.PROGRESS_STATUS_CRASHED)

        self.logger.info('Crash report stored : ' + self.git.git_last_commit)
        self.stop(JOB_STATUS.PROGRESS_STATUS_CRASHED)

    def write_log(self, message):
        if self.stream_log and self.running:
            self.stream_log.write(message)

    def set_status(self, status):
        self.logger.info('Job status changed to %s ' % (status,))
        self.job_add_status('status', status)

    @property
    def job_id(self):
        return self.git.job_id

    def create(self, create_info=None, hyperparameter=None, server='local', insights=None):
        """
        Creates a new job in git and pushes it.

        :param create_info: from the api.create_job_info(id). Contains e.g. code, layer information for simple models
        :param hyperparameter: simple nested dict with key->value.
        :param server:
        :param insights: whether you want to activate insights
        """
        self.job['server'] = server
        self.job['config'] = copy.deepcopy(self.config)
        self.job['optimization'] = None
        self.job['type'] = 'python'

        if 'parameters' not in self.job['config']:
            self.job['config']['parameters'] = {}

        if create_info is not None:
            if 'config' not in create_info:
                print(create_info)
                raise Exception('Given create_info is not valid.')

            self.job['type'] = create_info['type']
            self.job['config'].update(create_info['config'])

            if 'simple' == self.job['type']:
                for k, v in six.iteritems(create_info['datasets']):
                    if 'python' == v['type']:
                        self.git.add_file('aetros/dataset/' + k + '.py', v['code'])
                        del v['code']

                    self.git.add_file('aetros/dataset/' + k + '.json', json.dumps(v))

                self.git.add_file('aetros/model.py', json.dumps(create_info['code']))
                self.git.add_file('aetros/layer.json', json.dumps(create_info['layers']))

        if insights is not None:
            self.job['config']['insights'] = insights

        if hyperparameter is not None:
            self.job['config']['parameters'].update(hyperparameter)

        if 'insights' not in self.job['config']:
            self.job['config']['insights'] = False

        self.git.create_job_id(self.job)
        self.job['id'] = self.job_id

        self.logger.info("Job created " + self.model_name + '/' + self.job_id + " with git ref " + self.git.ref_head)
        return self.job_id


    def is_simple_model(self):
        if not self.job:
            raise Exception('Job not loaded yet. Use load(id) first.')

        if 'type' in self.job:
            return self.job['type'] == 'simple'

        return False

    def read_config(self, model_name=None):
        self.config = read_config(logger=self.logger)

        self.logger.debug('config: ' + json.dumps(self.config))

        if model_name is None and 'model' not in self.config:
            raise Exception('No AETROS Trainer model name given. Specify it in aetros.backend.start_job("model/name") or in .aetros.yml `model: model/name`.')

        if 'models' in self.config:
            if model_name in self.config['models']:
                self.logger.debug('Merged config values for ' + model_name)
                self.config.update(self.config['models'][model_name])

            del self.config['models']

        # todo, read parameters from script command arguments

        if not self.model_name and ('model' in self.job or not self.job['model']):
            raise Exception('No model name given. Specify in .aetros.yml or in aetros.backend.start_job("model/name")')

        ssh_command = self.config['ssh']
        ssh_command += ' -o StrictHostKeyChecking=no'

        if self.config['ssh_key']:
            ssh_command += ' -i "' + os.path.expanduser(self.config['ssh_key']) + '"'

        import tempfile
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(six.b(ssh_command + ' "$@"'))
        f.close()
        os.environ['GIT_SSH'] = f.name
        os.chmod(f.name, 0o700)

        import atexit
        def delelete_git_ssh_file():
            os.unlink(f.name)

        atexit.register(delelete_git_ssh_file)

        self.logger.debug('SSH_COMMAND:'+ssh_command)

    def get_parameter(self, path, default=None):
        """
        Reads hyper parameter from job configuration. If nothing found, fallback to user config in .aetros.yml
        :param path: str 
        :param default: *
        :return: *
        """
        params = self.job['config']['parameters']

        if path in params:
            return params[path]

        try:
            # try first parameters from job creator
            return self.read_dict_by_path(params, path)
        except:
            # if not found in parameters from the creator (Trainer usually) then try user config
            try:
                return self.read_dict_by_path(self.config['parameters'], path)
            except:
                if default is None:
                    raise

        # not found and default is not None
        return default

    def read_dict_by_path(self, dictionary, path):
        if path in dictionary:
            return dictionary[path]
        elif '.' not in path:
            raise Exception('Parameter ' + path + ' not found and no default value given.')

        if '.' in path:
            path = path.split('.')
            current = dictionary
            for item in path:
                if item not in current:
                    raise Exception('Parameter ' + path + ' not found and no default value given.')

                current = current[item]

            return current

    def load(self, job_id):
        """
        Loads job and sets as current.
        :param job_id: int
        """

        # normally git would create a job_id, but we have already one
        # so download the ref and check it out.
        self.git.fetch_job(job_id)

        if not os.path.exists(self.git.work_tree + '/aetros/job.json'):
            raise Exception('Could not load aetros/job.json from git repository. Make sure you have created the job correctly.')

        with open(self.git.work_tree + '/aetros/job.json') as f:
            self.job = json.loads(f.read())

        print(str(self.job))

        if not self.job:
            raise Exception('Could not parse aetros/job.json from git repository. Make sure you have created the job correctly.')

        self.job['id'] = job_id

        if os.path.exists(self.git.work_tree + '/aetros/job/status/progress.json'):
            with open(self.git.work_tree + '/aetros/job/status/progress.json', 'r') as f:
                progress = float(f.read().decode('utf-8'))
        else:
            progress = 0

        if progress >= 2:
            self.logger.error('You can not restart an existing job that was already running. You need to restart the job through AETROS Trainer.')
            self.logger.error('You can alternatively reset the git reference to the root commit and force push that ref to reset the job.')
            sys.exit(1)

        self.logger.debug(str(self.job))

    def get_job_model(self):
        """
        Returns a new JobModel instance with current loaded job data attached.
        :return: JobModel
        """
        if not self.job:
            raise Exception('Job not loaded yet. Use load(id) first.')

        from aetros.JobModel import JobModel

        return JobModel(self.job_id, self.job, self.config['storage_dir'])

    def sync_weights(self):
        self.job_add_status('status', 'SYNC WEIGHTS')
        print("Sync weights ...")
        try:
            self.upload_weights('latest.hdf5', self.get_job_model().get_weights_filepath_latest(), with_status=True)
        except:
            pass
        print("Weights synced.")

    def job_add_status(self, key, value):
        self.git.store_file('aetros/job/status/' + key + '.json', json.dumps(value, default=invalid_json_values))

    def set_info(self, key, value, commit_end_of_job=False):
        if commit_end_of_job:
            self.git.store_file('aetros/job/info/' + key + '.json', json.dumps(value, default=invalid_json_values))
        else:
            self.git.commit_json_file('INFO ' + key, 'aetros/job/info/' + key, value)

    def set_graph(self, graph):
        self.git.commit_json_file('GRAPH', 'aetros/job/graph', graph)

    def set_system_info(self, key, value, commit_end_of_job=False):
        if commit_end_of_job:
            self.git.store_file('aetros/job/system/' + key + '.json', json.dumps(value, default=invalid_json_values))
        else:
            self.git.commit_json_file('SYSTEM_INFO ' + key, 'aetros/job/system/' + key, value)

    def commit_file(self, path, title=None):
        path = os.path.expanduser(path).strip()

        path = os.path.relpath(path, os.getcwd())
        if path.startswith('..'):
            raise Exception('Can not commit files that are out of the current working directory.')

        if './' == path[0: 2]:
            path = path[2:]

        with self.git.batch_commit('FILE ' + (title or path)):
            if os.path.isdir(path):
                for file in os.listdir(path):
                    self.commit_file(path + '/' + file)
                return

            if os.path.getsize(path) > 10 * 1024 * 1024 * 1024:
                raise Exception('Can not upload file bigger than 10MB')

            with open(path, 'rb') as f:
                contents = f.read()

            self.git.commit_file('FILE ' + (title or path), path, contents)

    def job_add_insight(self, x, images, confusion_matrix):
        converted_images = []
        info = {}
        for image in images:
            if not isinstance(image, JobImage):
                raise Exception('job_add_insight only accepts JobImage instances in images argument')

            converted_images.append({
                'id': image.id,
                'image': self.pil_image_to_jpeg(image.image)
            })
            info[image.id] = {
                'file': image.id+'.jpg',
                'label': image.label
            }

        with self.git.batch_commit('INSIGHT ' + str(x)):

            for image in converted_images:
                self.git.commit_file('INSIGHT ' + str(image['id']), 'aetros/job/insight/'+str(x)+'/image/'+image['id']+'.jpg', image['image'])

            self.git.commit_json_file('INSIGHT CONFUSION_MATRIX', 'aetros/job/insight/'+str(x)+'/confusion_matrix', confusion_matrix)
            self.git.commit_json_file('INSIGHT INFO', 'aetros/job/insight/' + str(x) + '/info', info)

    def pil_image_to_jpeg(self, image):
        buffer = six.BytesIO()

        image.save(buffer, format="JPEG", optimize=True, quality=70)
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
