from __future__ import absolute_import
from __future__ import print_function

import atexit
import inspect
import os
import socket
import traceback

from threading import Thread, Lock

import collections
import coloredlogs
import logging

import math
import requests
import signal
import time

import simplejson
import six
import PIL.Image
import sys
import msgpack
from ruamel import yaml
from ruamel.yaml.reader import ReaderError

from aetros.JobModel import JobModel
from aetros.const import JOB_STATUS, __version__
from aetros.cuda_gpu import CudaNotImplementedException
from aetros.git import Git
from aetros.logger import GeneralLogger
from aetros.utils import git, invalid_json_values, read_config, is_ignored, prepend_signal_handler, raise_sigint, \
    read_parameter_by_path, stop_time, read_home_config, lose_parameters_to_full, extract_parameters, create_ssh_stream
from aetros.MonitorThread import MonitoringThread
import subprocess

if not isinstance(sys.stdout, GeneralLogger):
    sys.stdout = GeneralLogger(redirect_to=sys.__stdout__)

if not isinstance(sys.stderr, GeneralLogger):
    sys.stderr = GeneralLogger(redirect_to=sys.__stderr__)

last_exit_code = None
original_exit = sys.exit


def patched_exit(status=None):
    global last_exit_code
    last_exit_code = status
    original_exit(status)

sys.exit = patched_exit


def on_shutdown():
    for job in on_shutdown.started_jobs:
        job.on_shutdown()

on_shutdown.started_jobs = []

atexit.register(on_shutdown)


def Popen(*args, **kwargs):
    """
    Executes a command using subprocess.Popen and redirects output to AETROS and stdout.
    Parses stdout as well for stdout API calls.

    Use read_line argument to read stdout of command's stdout line by line.
    Use returned process stdin to communicate with the command.

    :return: subprocess.Popen
    """

    read_line = None
    if 'read_line' in kwargs:
        read_line = kwargs['read_line']
        del kwargs['read_line']

    p = subprocess.Popen(*args, **kwargs)
    wait_stdout = None
    wait_stderr = None

    if p.stdout:
        wait_stdout = sys.stdout.attach(p.stdout, read_line=read_line)
    if p.stderr:
        wait_stderr = sys.stderr.attach(p.stderr)

    original_wait = p.wait
    def wait():
        original_wait()

        if wait_stdout:
            wait_stdout()
        if wait_stderr:
            wait_stderr()

    p.wait = wait

    return p

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
            kwargs['json'] = simplejson.loads(simplejson.dumps(json_chunk, default=invalid_json_values), object_pairs_hook=collections.OrderedDict)

        return requests.get(self.get_url(url), params=params, **kwargs)

    def post(self, url, data=None, **kwargs):
        json_chunk = kwargs.get('json')
        if (json_chunk and not isinstance(json_chunk, str)):
            kwargs['json'] = simplejson.loads(simplejson.dumps(json_chunk, default=invalid_json_values), object_pairs_hook=collections.OrderedDict)

        return requests.post(self.get_url(url), data=data, **kwargs)

    def put(self, url, data=None, **kwargs):
        json_chunk = kwargs.get('json')
        if (json_chunk and not isinstance(json_chunk, str)):
            kwargs['json'] = simplejson.loads(simplejson.dumps(json_chunk, default=invalid_json_values), object_pairs_hook=collections.OrderedDict)

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
        self.config = config
        self.host = config['host']
        self.ssh_stream = None
        self.ssh_stream_stdout = None
        self.ssh_stream_stdin = None
        self.go_offline_on_first_failed_attempt = True

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
        self.stop_on_empty_queue = False

        # indicates whether we are offline or not, means not connected to the internet and
        # should not establish a connection to Aetros.
        self.online = True

        # Whether the client is active and should do things.
        self.active = False
        self.expect_close = False
        self.external_stopped = False

        # the connection is authenticated against the server and ready to send stuff
        self.registered = False

        # the actual connection is established
        self.connected = False

        self.was_connected_once = False
        self.read_unpacker = msgpack.Unpacker(encoding='utf-8')

    def on_sigint(self, sig, frame):
        # when connections breaks, we do not reconnect
        self.expect_close = True

    def start(self):
        self.active = True

        prepend_signal_handler(signal.SIGINT, self.on_sigint)

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
        Also, when more than 10 connection tries are detected, we delay extra 15 seconds.
        """
        if self.connection_tries > 10:
            time.sleep(15)

        if self.in_connecting:
            return False

        self.in_connecting = True

        self.logger.debug('Wanna connect ...')

        try:
            if self.connected or not self.online:
                return True

            self.ssh_stream = create_ssh_stream(self.config, exit_on_failure=False)
            self.ssh_stream_stdin, self.ssh_stream_stdout, stderr = self.ssh_stream.exec_command('stream')

            self.logger.debug('Open ssh')
            messages = self.wait_for_at_least_one_message()
            stderrdata = ''

            if not messages:
                stderrdata = stderr.read().decode("utf-8").strip()
            else:
                self.connected = True
                self.registered = self.on_connect(self.was_connected_once)

            if not self.registered:
                self.logger.debug("Client: registration failed. stderrdata: " + stderrdata)
                self.connected = False

                try:
                    self.logger.debug('Client: ssh_tream close')
                    self.ssh_stream.close()
                except (KeyboardInterrupt, SystemExit): raise
                except Exception: pass

                self.connection_tries += 1
                if not self.was_connected_once and self.go_offline_on_first_failed_attempt:
                    # initial try needs to be online, otherwise we go offline
                    self.go_offline()

                if stderrdata:
                    if 'Connection refused' not in stderrdata and 'Permission denied' not in stderrdata:
                        self.logger.error(stderrdata)

                if 'Permission denied' in stderrdata:
                    if self.connection_tries < 3:
                        self.logger.warning("Access denied. Did you setup your SSH public key correctly and saved it in your AETROS Trainer user account?")

                    self.close()
                    sys.exit(1)

                self.connection_error("Connection error during connecting to %s: %s" % (self.host, str(stderrdata)))
            else:
                self.was_connected_once = True

        except Exception as error:
            self.connection_error(error)
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
            # we don't care when we're not active
            return

        # give is some free time
        time.sleep(0.1)

        # make sure ssh connection is closed, so we can recover
        try:
            if self.ssh_stream:
                self.logger.debug('Client: ssh_tream close')
                self.ssh_stream.close()
        except (KeyboardInterrupt, SystemExit): raise
        except Exception: pass

        if self.expect_close:
            # we expected the close, so ignore the error
            return

        # needs to be set before logger.error, since they can call send_message again
        self.connected = False
        self.registered = False

        if socket is None:
            # python interpreter is already dying, so quit
            return

        message = "Connection error"

        if error:
            import traceback
            sys.stderr.write(traceback.format_exc() + '\n')

            if hasattr(error, 'message'):
                self.logger.error(message + ": " + str(error.message))
            else:
                self.logger.error(message + ": " + str(error))

            if 'No authentication methods available' in str(error):
                self.logger.error("Make sure you have authenticated your machine correctly using 'aetros authenticate'.")
        else:
            self.logger.error(message)

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
                                    # not too much at once (max 1MB), so we have time to listen for incoming messages
                                    if sent_size > 1024 * 1024:
                                        break
                                else:
                                    break

                        self.lock.acquire()
                        for message in sent:
                            if message in self.queue:
                                self.queue.remove(message)
                        self.lock.release()

                        if self.stop_on_empty_queue:
                            self.logger.debug('Client sent %d / %d messages' % (len(sent), len(self.queue)))
                            return
                    except Exception as e:
                        self.logger.debug('Closed write thread: exception. %d messages left' % (len(self.queue), ))
                        self.connection_error(e)

                if self.active and not self.connected and not self.expect_close:
                    if not self.connect():
                        time.sleep(5)

            time.sleep(0.1)

        self.logger.debug('Closed write thread: ended. %d messages left' % (len(self.queue), ))

    def thread_read(self):
        while self.active:
            if self.online:
                if self.connected and self.registered:
                    try:
                        messages = self.read() # this blocks

                        if messages is not None:
                            self.handle_messages(messages)

                        continue
                    except Exception as e:
                        self.logger.debug('Closed read thread: exception')
                        self.connection_error(e)

            time.sleep(0.01)

        self.logger.debug('Closed read thread: ended')

    def wait_sending_last_messages(self):
        if self.active and self.online and self.connected and self.registered:
            # send all missing messages
            self.stop_on_empty_queue = True
            self.thread_write_instance.join()

    def wait_for_close(self):
        if not (self.active and self.online and self.connected and self.registered):
            return

        self.active = False

        i = 0
        try:
            while self.ssh_stream_stdout and self.ssh_stream_stdout.read() != six.b(''):
                i += 1
                time.sleep(0.1)
                if i % 50 == 0:
                    self.logger.warning("We are still waiting for connection closing on server side.")
        except Exception: pass

        self.online = False

    def close(self):
        self.active = False
        self.connected = False

        if self.ssh_stream:
            try:
                self.logger.debug('Client: ssh_tream close')
                self.ssh_stream.close()
            except (KeyboardInterrupt, SystemExit): raise
            except Exception: pass

        if self.online:
            self.event_listener.fire('close')

        self.online = False

    def send(self, message):
        if not (self.active and self.online):
            # It's important to queue anything when active and online
            # as we would lose information in git streams.
            return

        if self.stop_on_empty_queue:
            # make sure, we don't add new one
            return

        self.message_id += 1
        message['_id'] = self.message_id
        message['_sending'] = False
        message['_sent'] = False

        self.queue.append(message)

    def send_message(self, message):
        """
        Internal. Sends the actual message from a queue entry.
        """
        if not self.connected:
            return False

        message['_sending'] = True

        msg = msgpack.packb(message, default=invalid_json_values)

        try:
            self.ssh_stream_stdin.write(msg)
            message['_sent'] = True
            self.ssh_stream_stdin.flush()

            return len(msg)
        except (KeyboardInterrupt, SystemExit):

            if message['_sent']:
                return len(msg)

            return False

        except Exception as error:
            self.connection_error(error)
            return False

    def handle_messages(self, messages):
        for message in messages:
            if not isinstance(message, dict): continue

            if 'a' in message:
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
                chunk = self.ssh_stream_stdout.read(1)
            except Exception as error:
                self.connection_error(error)
                raise

            if chunk == '':
                # happens only when connection broke. If nothing is to be received, it hangs instead.
                self.connection_error('Connection broken w')
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
            chunk = self.ssh_stream_stdout.read(1)
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
    def configure(self, model_name, job_id, name):
        self.model_name = model_name
        self.job_id = job_id
        self.name = name

    def on_connect(self, reconnect=False):
        self.send_message({
            'type': 'register_job_worker',
            'model': self.model_name,
            'job': self.job_id,
            'reconnect': reconnect,
            'version': __version__,
            'name': self.name
        })

        self.logger.debug("Wait for job client registration for " + self.name)
        messages = self.wait_for_at_least_one_message()
        self.logger.debug("Got " + str(messages))

        if not messages:
            self.event_listener.fire('registration_failed', {'reason': 'No answer received.'})
            return False

        message = messages.pop(0)
        self.logger.debug("Client: handle message: " + str(message))
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

    def handle_messages(self, messages):
        BackendClient.handle_messages(self, messages)
        for message in messages:
            if self.external_stopped:
                continue

            if not isinstance(message, dict): continue

            if 'a' in message and 'parameter-changed' == message['a']:
                self.event_listener.fire('parameter_changed', {'values': message['values']})

            if 'type' in message and 'action' == message['type']:
                self.event_listener.fire('action', message)


def context():
    """
    Returns a new JobBackend instance which connects to AETROS Trainer
    based on "model" in aetros.yml or env:AETROS_MODEL_NAME environment variable.

    If env:AETROS_JOB_ID is not defined, it creates a new job.

    Job is ended either by calling JobBackend.done(), JobBackend.fail() or JobBackend.abort().
    If the script ends without calling one of the methods above, JobBackend.done is automatically called.

    :return: JobBackend
    """
    job = JobBackend()

    if os.getenv('AETROS_JOB_ID'):
        job.load(os.getenv('AETROS_JOB_ID'))
    else:
        job.create()

    job.start()

    return job


def start_job(name=None):
    """
    Tries to load the job defined in the AETROS_JOB_ID environment variable.
    If not defined, it creates a new job.
    Starts the job as well.

    :param name: string: model name
    :return: JobBackend
    """

    job = JobBackend(name)

    if os.getenv('AETROS_JOB_ID'):
        job.load(os.getenv('AETROS_JOB_ID'))
    else:
        job.create()

    job.start()

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
        self.lock = Lock()

        self.job_backend.git.commit_json_file('CREATE_CHANNEL', 'aetros/job/channel/' + name+ '/config', message)
        self.stream = self.job_backend.git.stream_file('aetros/job/channel/' + name+ '/data.csv')
        self.stream.write('"x","training","validation"\n')

    def send(self, x, training_loss, validation_loss):
        line = simplejson.dumps([x, training_loss, validation_loss])[1:-1]

        self.lock.acquire()
        try:
            self.stream.write(line + "\n")
            self.job_backend.git.store_file('aetros/job/channel/' + self.name + '/last.csv', line)
        finally:
            self.lock.release()


class JobImage:
    def __init__(self, id, image, label=None, pos=None):
        self.id = id
        if not isinstance(image, PIL.Image.Image):
            raise Exception('JobImage requires a PIL.Image as image argument.')

        self.image = image
        self.label = label
        self.pos = pos


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
        self.lock = Lock()

        if self.kpi:
            self.job_backend.kpi_channel = self

        if not (isinstance(traces, list) or traces is None):
            raise Exception(
                'traces can only be None or a list of dicts: [{name: "name", option1: ...}, {name: "name2"}, ...]')

        if not traces:
            traces = [{'name': ''}]

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

        line = simplejson.dumps(['x'] + [str(x['name']) for x in traces])[1:-1]
        self.stream.write(line + "\n")

    def send(self, x, y):
        if not isinstance(y, list):
            y = [y]

        if len(y) != len(self.traces):
            raise Exception(
                'You tried to set more y values (%d items) then traces available in channel %s (%d traces).' % (
                    len(y), self.name, len(self.traces)))

        for v in y:
            if not isinstance(v, (int, float)) and not isinstance(v, six.string_types):
                raise Exception('Could not send channel value for ' + self.name+' since type ' + type(y).__name__+' is not supported. Use int, float or string values.')

        line = simplejson.dumps([x] + y)[1:-1]

        self.lock.acquire()
        try:
            self.stream.write(line + "\n")
            self.job_backend.git.store_file('aetros/job/channel/' + self.name + '/last.csv', line)
        finally:
            self.lock.release()

        if self.kpi:
            self.job_backend.git.store_file('aetros/job/kpi/last.json', simplejson.dumps(y[self.kpiTrace]))


class JobBackend:
    """
    :type event_listener: EventListener
    :type id: str: model name
    :type job_id: str: job id
    :type client: Client
    :type job: dict
    """

    def __init__(self, model_name=None, logger=None, config_path = 'aetros.yml', name=None):
        self.event_listener = EventListener()

        self.log_file_handle = None
        self.job = {'parameters': {}}

        self.git = None

        self.ssh_stream = None
        self.model_name = model_name
        self.logger = logger
        self.config_path = config_path

        self.name = name
        if not self.name:
            if os.getenv('AETROS_JOB_NAME'):
                self.name = os.getenv('AETROS_JOB_NAME')
            else:
                self.name = 'master'

        self.client = None
        self.stream_log = None
        self.speed_stream = None
        self.parameter_change_callback = None
        self.lock = Lock()

        self.last_speed = 0
        self.last_speed_time = 0
        self.last_step_time = time.time()
        self.last_step = 0
        self.made_steps_since_last_sync = 0
        self.made_steps_size_since_last_sync = 0

        self.start_time = time.time()
        self.current_epoch = 0
        self.total_epochs = 0

        self.step_label = None
        self.step_speed_label = None

        # indicates whether early_stop() has been called. Is called by reaching maxTimes or maxEpochs limitation.
        # This flag stops exiting with > 0, since the reach of a limitation is a valid exit.
        self.in_early_stop = False

        # ended means: done, abort or fail method has been called.
        self.ended = False

        # when stop(wait_for_client=True) is called, we sync last messages.
        # this flag indicates that end() hasn't been called yet
        self.stopped = False

        # running means: the syncer client is running.
        self.running = False

        # whether it has started once
        self.started = False

        # whether we are in paused state
        self.is_paused = False

        # whether on_shutdown has been called and thus the python interpreter is dying.
        self.in_shutdown = False

        self.monitoring_thread = None

        if not self.logger:
            self.logger = logging.getLogger('aetros-job')
            atty = None
            if '1' == os.getenv('AETROS_ATTY'):
                atty = True
            coloredlogs.install(level=self.log_level, logger=self.logger, isatty=atty)

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
        self.event_listener.on('parameter_changed', self.on_parameter_changed)
        self.event_listener.on('action', self.on_action)

        if hasattr(signal, 'SIGUSR1'):
            prepend_signal_handler(signal.SIGUSR1, self.on_signusr1)

        self.pid = os.getpid()

        self.ensure_model_name()
        self.home_config = read_home_config()
        self.client = JobClient(self.home_config, self.event_listener, self.logger)
        self.git = Git(self.logger, self.client, self.home_config, self.model_name)

        self.logger.debug("Started tracking of job files in git %s for remote %s" % (self.git.git_path, self.git.origin_url))

    @property
    def log_level(self):
        if os.getenv('DEBUG') == '1':
            return 'DEBUG'

        return 'INFO'

    @property
    def host(self):
        return self.home_config['host']

    def section(self, title):
        title = title.replace("\t", "  ")
        sys.stdout.write("\f" + title+ "\t" + str(time.time() - self.start_time) + "\n")
        sys.stdout.flush()

    def on_registration_failed(self, params):
        self.logger.warning("Connecting to AETROS Trainer at %s failed. Reasons: %s" % (self.host, params['reason'],))
        if 'Permission denied' in params['reason']:
            self.logger.warning("Make sure you have saved your ssh pub key in your AETROS Trainer user account.")

    def on_client_offline(self, params):
        if self.is_master_process():
            self.logger.warning("Could not establish a connection. We stopped automatic syncing.")
            self.logger.warning("You can publish later this job to AETROS Trainer by executing following command.")
            self.logger.warning("$ aetros push-job " + self.model_name + "/" + self.job_id)

        self.git.online = False

    def on_registration(self, params):
        if not self.registered:
            self.registered = True
            self.logger.debug('Git backend start')
            self.git.start()
        else:
            self.logger.info("Successfully reconnected.")

    def on_signusr1(self, signal, frame):
        self.logger.warning("USR1: backend job_id=%s (running=%s, ended=%s), client (online=%s, active=%s, registered=%s, "
                            "connected=%s, queue=%d), git (online=%s, active_thread=%s, last_push_time=%s)." % (
          str(self.job_id),
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

    def on_force_exit(self):
        """
        External hook.
        """
        pass

    def on_sigint(self, sig, frame):
        """
        We got SIGINT signal.
        """

        if self.stop_requested or self.stop_requested_force:
            # signal has already been sent or we force a shutdown.
            # handles the keystroke 2x CTRL+C to force an exit.
            self.stop_requested_force = True
            self.logger.warning('Force stopped: ' + str(sig))

            # just kill the process, we don't care about the results
            os._exit(1)
            # with force_exit we really close the process, killing it in unknown state
            # self.fail('Force stopped', force_exit=True)
            # return

        if self.is_master_process():
            self.logger.warning('Received '+str(sig)+' signal. Send again to force stop. Stopping ...')
        else:
            sys.__stdout__.write("Got child signal " + str(sig) +"\n")
            sys.__stdout__.flush()

        self.stop_requested = True

        # the default SIGINT handle in python is not always installed, so we can't rely on the
        # KeyboardInterrupt exception to be thrown.
        # thread.interrupt_main would call sigint again.
        # the shutdown listener will do the rest like committing rest memory files into Git and closing connections.
        sys.exit(0 if self.in_early_stop else 1)

    def external_aborted(self, params):
        """
        Immediately abort the job by server.

        This runs in the Client:read() thread.
        """
        self.ended = True
        self.running = False

        # When the server sends an abort signal, we really have to close immediately,
        # since for example the job has been already deleted.
        # without touching the git and client any further
        os._exit(1)

    def external_stop(self, force):
        """
        Stop signal by server.
        """

        # only the master processes handles the regular stop signal from the server, sending a SIGINT to
        # all its child (means to us, non-master process)
        if not self.is_master_process():
            if force:
                # make sure even the subprocess dies really on force
                os._exit(1)

            return

        self.logger.warning("Received stop signal by server.")
        if not self.stop_requested_force:
            self.stop_requested_force = force

        raise_sigint()

    def early_stop(self):
        """
        Stop when a limitation is reached (like maxEpoch, maxTime).
        """
        self.in_early_stop = True

        raise_sigint()

    def batch(self, batch, total, batch_size, label='BATCH', speed_label='SAMPLES/S'):
        self.step(batch, total, size=batch_size, label=label, speed_label=speed_label)

    def sample(self, sample, total, label='SAMPLE', speed_label='SAMPLES/S'):
        self.step(sample, total, size=1, label=label, speed_label=speed_label)

    def step(self, step, total, label='STEP', speed_label='STEPS/S', size=1):
        """
        Increase the step indicator, which is a sub progress circle of the actual
        main progress circle (epoch, progress() method).
        """

        self.lock.acquire()
        try:
            time_diff = time.time() - self.last_step_time
            # print("steps %s/%s, size=%s, last_step=%s, made_steps=%s" % (str(step), str(total), str(size), str(self.last_step), str(step - self.last_step)))

            if self.last_step > step:
                # it restarted
                self.last_step = 0

            made_steps_since_last_call = step - self.last_step
            self.last_step = step

            self.made_steps_since_last_sync += made_steps_since_last_call
            self.made_steps_size_since_last_sync += made_steps_since_last_call * size

            if time_diff >= 1 or step == total:  # only each second or last batch
                self.set_system_info('step', step, True)
                self.set_system_info('steps', total, True)

                steps_per_second = self.made_steps_since_last_sync / time_diff
                samples_per_second = self.made_steps_size_since_last_sync / time_diff
                self.last_step_time = time.time()

                # print("  samples_per_second=%s, steps_per_second=%s, made_steps=%s, time_diff=%s, " %
                #       (str(samples_per_second), str(steps_per_second), str(self.made_steps_since_last_sync), str(time_diff)))

                if size:
                    self.report_speed(samples_per_second)

                epochs_per_second = steps_per_second / total  # all batches
                self.set_system_info('epochsPerSecond', epochs_per_second, True)

                current_epochs = self.current_epoch if self.current_epoch else 1
                total_epochs = self.total_epochs if self.total_epochs else 1

                self.made_steps_since_last_sync = 0
                self.made_steps_size_since_last_sync = 0

                eta = 0
                if step < total:
                    # time to end this epoch
                    if steps_per_second != 0:
                        eta = (total - step) / steps_per_second

                # time until all epochs are done
                if total_epochs - current_epochs > 0:
                    if epochs_per_second != 0:
                        eta += (total_epochs - (current_epochs)) / epochs_per_second

                self.git.store_file('aetros/job/times/eta.json', simplejson.dumps(eta))

            if label and self.step_label != label:
                self.set_system_info('stepLabel', label, True)
                self.step_label = label

            if speed_label and self.step_speed_label != speed_label:
                self.set_system_info('stepSpeedLabel', speed_label, True)
                self.step_speed_label = speed_label
        finally:
            self.lock.release()

    def report_speed(self, speed, x=None, label=None):
        if not self.is_master_process():
            self.stdout_api_call('speed', x=x, speed=speed, label=label)
            return

        self.last_speed = speed
        self.last_speed_time = time.time()

        if x is None:
            x = math.ceil(time.time()-self.start_time)

        self.set_system_info('samplesPerSecond', speed, True)
        self.speed_stream.write(simplejson.dumps([x, speed])[1:-1] + "\n")

        if label and self.step_speed_label != label:
            self.set_system_info('stepSpeedLabel', label, True)
            self.step_speed_label = label

    def stdout_api_call(self, command, **kwargs):
        action = {'aetros': command}
        action.update(kwargs)
        print(simplejson.dumps(action))

    @property
    def job_settings(self):
        if 'settings' in self.job['config']:
            return self.job['config']['settings']
        return {}

    def set_parameter_change_callback(self, callback):
        self.parameter_change_callback = callback

    def on_parameter_changed(self, params):
        pass

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
                self.git.store_file('aetros/job/progress/' + self.id + '.json', simplejson.dumps(info))

            def label(self, label):
                self._label = label
                self.store()

            def start(self):
                if self.started is not False:
                    return

                self.step = 0
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

    def epoch(self, epoch=None, total=None):
        self.progress(epoch, total)

    def progress(self, epoch=None, total=None):
        self.current_epoch = self.current_epoch if epoch is None else epoch
        self.total_epochs = self.total_epochs if total is None else total
        epoch_limit = False

        if 'maxEpochs' in self.job['config'] and isinstance(self.job['config']['maxEpochs'], int) and self.job['config']['maxEpochs'] > 0:
            epoch_limit = True
            self.total_epochs = self.job['config']['maxEpochs']

        if self.current_epoch is not 0 and self.last_progress_call:
            # how long took it since the last call?
            time_per_epoch = time.time() - self.last_progress_call
            eta = time_per_epoch * (self.total_epochs - self.current_epoch)
            if self.current_epoch > self.total_epochs:
                eta = 0

            self.git.store_file('aetros/job/times/eta.json', simplejson.dumps(eta))

            if time_per_epoch > 0:
                self.set_system_info('epochsPerSecond', 1 / time_per_epoch, True)

        self.set_system_info('epoch', self.current_epoch, True)
        self.set_system_info('epochs', self.total_epochs, True)
        self.last_progress_call = time.time()

        if epoch_limit and self.total_epochs > 0:
            if self.current_epoch >= self.total_epochs:
                self.logger.warning("Max epoch of "+str(self.total_epochs)+" reached")
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

    def start(self, collect_system=True):
        if self.started:
            raise Exception('Job was already started.')

        if self.running:
            raise Exception('Job already running.')

        if not self.job_id:
            raise Exception('No job id found. Use create() first.')

        if not self.job:
            raise Exception('Job not loaded')

        prepend_signal_handler(signal.SIGINT, self.on_sigint)

        self.started = True
        self.running = True
        self.ended = False

        on_shutdown.started_jobs.append(self)

        self.client.configure(self.model_name, self.job_id, self.name)
        self.git.prepare_git_user()

        if self.git.online:
            self.logger.debug('Job backend start')
            self.client.start()
        else:
            self.logger.debug('Job backend not started, since being online not detected.')

        if self.is_master_process():
            # this is the process that actually starts the job.
            # other sub-processes may only modify other data.
            self.git.commit_file('JOB_STARTED', 'aetros/job/times/started.json', str(time.time()))
            self.job_add_status('progress', JOB_STATUS.PROGRESS_STATUS_STARTED)
            self.git.store_file('aetros/job/times/elapsed.json', str(0))

            if collect_system:
                self.collect_system_information()
                self.collect_environment()

            # make sure we get the progress first, before monitoring sends elapses and
            # updates the job cache
            if self.git.online:
                self.git.push()

            self.logger.info("Job %s/%s started." % (self.model_name, self.job_id))
            self.logger.info("Open http://%s/model/%s/job/%s to monitor it." % (self.host, self.model_name, self.job_id))

            if collect_system:
                self.start_monitoring()

            # log stdout to Git by using self.write_log -> git:stream_file
            self.stream_log = self.git.stream_file('aetros/job/log.txt')
            self.speed_stream = self.git.stream_file('aetros/job/speed.csv')
            header = ["x", "speed"]
            self.speed_stream.write(simplejson.dumps(header)[1:-1] + "\n")

            if isinstance(sys.stdout, GeneralLogger):
                sys.stdout.job_backend = self
                sys.stdout.flush()

            if isinstance(sys.stderr, GeneralLogger):
                sys.stderr.job_backend = self
                sys.stdout.flush()
        else:
            # if this process has been called within another process that is already using JobBackend.
            # we disable some stuff
            if isinstance(sys.stdout, GeneralLogger) and not sys.stderr.job_backend:
                sys.stdout.disable_buffer()

            if isinstance(sys.stderr, GeneralLogger) and not sys.stderr.job_backend:
                sys.stderr.disable_buffer()

    def set_paused(self, v):
        self.is_paused = v
        self.set_system_info('paused', self.is_paused, True)

    def is_master_process(self):
        """
        Master means that aetros.backend.start_job() has been called without using the command `aetros start`.
        If master is true, we collect and track some data that usually `aetros start` would do and reset the job's
        temp files on the server.
        :return:
        """

        return os.getenv('AETROS_JOB_ID') is None

    def detect_git_version(self, working_dir=None):
        current_dir = os.getcwd()

        try:
            if working_dir:
                os.chdir(working_dir)

            with self.git.batch_commit('Git Version'):
                value = git.get_current_remote_url()
                if value:
                    self.set_system_info('git_remote_url', value)

                value = git.get_current_commit_hash()
                if value:
                    self.set_system_info('git_version', value)

                value = git.get_current_branch()
                if value:
                    self.set_system_info('git_branch', value)

                value = git.get_current_commit_message()
                if value:
                    self.set_system_info('git_commit_message', value)

                value = git.get_current_commit_author()
                if value:
                    self.set_system_info('git_commit_author', value)
        finally:
            if working_dir:
                os.chdir(current_dir)

    def start_monitoring(self, start_time=None, cpu_cores=1, gpu_devices=None, docker_container=None):
        if not self.monitoring_thread:
            self.monitoring_thread = MonitoringThread(self, cpu_cores, start_time, gpu_devices, docker_container)
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

        from aetros.KerasCallback import KerasCallback
        self.callback = KerasCallback(self, self.logger, force_insights=insights)
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
        """
        Shutdown routine. Sets the last progress (done, aborted, failed) and tries to send last logs and git commits.
        Also makes sure the ssh connection is closed (thus, the job marked as offline).

        Is triggered by atexit.register().
        """

        self.in_shutdown = True

        if self.stopped or self.ended:
            # make really sure, ssh connection closed
            self.client.close()
            return

        if self.in_early_stop:
            self.done()
            return

        if self.stop_requested:
            # when SIGINT has been triggered
            if self.stop_requested_force:
                if not self.is_master_process():
                    # if not master process, we just stop everything. status/progress is set by master
                    self.stop(wait_for_client=True, force_exit=True)
                else:
                    # master process
                    self.fail('Force stopped.', force_exit=True)
            else:
                if not self.is_master_process():
                    # if not master process, we just stop everything. status/progress is set by master
                    self.stop(wait_for_client=True)
                else:
                    # master process
                    self.abort()

            return

        if hasattr(sys, 'last_value'):
            # sys.last_value contains a exception, when there was an uncaught one
            if isinstance(sys.last_value, KeyboardInterrupt):
                # can only happen when KeyboardInterrupt has been raised manually
                # since the one from the default sigint handler will never reach here
                # since we catch the sigint signal and sys.exit() before the default sigint handler
                # is able to raise KeyboardInterrupt
                self.abort()
            else:
                self.fail(type(sys.last_value).__name__ + ': ' + str(sys.last_value))

        elif self.running:
            self.done()

    def done(self, force_exit=False):
        if not self.running:
            return

        self.stop(JOB_STATUS.PROGRESS_STATUS_DONE, wait_for_client=True, force_exit=force_exit)

    def send_std_buffer(self):
        if isinstance(sys.stdout, GeneralLogger):
            sys.stdout.send_buffer()

        if isinstance(sys.stderr, GeneralLogger):
            sys.stderr.send_buffer()

    def stop(self, progress=None, wait_for_client=False, force_exit=False):
        global last_exit_code

        if self.stopped:
            return

        if self.is_master_process():
            self.section('end')

        if self.monitoring_thread:
            self.monitoring_thread.stop()

        self.logger.debug("stop: " + str(progress))

        self.send_std_buffer()

        self.stopped = True
        self.ended = True
        self.running = False

        if self.git.online and not force_exit:
            if wait_for_client:
                self.logger.debug("client sends last %d messages ..." % (len(self.client.queue),))
                self.client.wait_sending_last_messages()

        # store all store_file and stream_file as blob
        self.logger.debug("Git stopping ...")
        self.git.stop()

        was_online = self.client.online

        if self.client.online and not force_exit:
            # server stores now all store_file and stream_file as blob as well,
            # wait for connection close to make sure server stored the files in server's git.
            # This happens after self.client.wait_sending_last_message() because it makes sure,
            # the server received all git streaming files.
            self.logger.debug("client stopping ...")
            self.client.end()

        self.client.close()

        exit_code = last_exit_code or 0
        if progress == JOB_STATUS.PROGRESS_STATUS_DONE:
            exit_code = 0

        if progress == JOB_STATUS.PROGRESS_STATUS_ABORTED:
            exit_code = 1

        if progress == JOB_STATUS.PROGRESS_STATUS_FAILED:
            exit_code = 2

        if self.is_master_process():
            self.set_system_info('exit_code', exit_code)

        # everything stopped. Server has all blobs, so we add the last progress
        # and do last git push.
        if self.is_master_process() and progress is not None:
            # if not master process, the master process will set it
            self.git.commit_file('STOP', 'aetros/job/status/progress.json', simplejson.dumps(progress, default=invalid_json_values))

        if not force_exit and was_online and self.is_master_process():
            # both sides should have same blobs, so we do now our final push
            self.logger.debug("git last push ...")
            successful_push = self.git.push()

            if not successful_push:
                self.logger.warning("Not all job information has been uploaded.")
                self.logger.warning("Please run following command to make sure your job is stored on the server.")
                self.logger.warning("$ aetros push-job " + self.model_name + "/" + self.job_id)

        if self.is_master_process():
            # remove the index file
            self.git.clean_up()

        self.logger.debug("Stopped %s with last commit %s." % (self.git.ref_head, self.git.git_last_commit))

        if force_exit:
            self.on_force_exit()
            os._exit(exit_code)
        elif not self.in_shutdown:
            sys.exit(exit_code)

    def abort(self, wait_for_client_messages=True, force_exit=False):
        if not self.running:
            return

        self.set_status('ABORTED', add_section=False)

        self.stop(JOB_STATUS.PROGRESS_STATUS_ABORTED, wait_for_client_messages, force_exit=force_exit)

    def fail(self, message=None, force_exit=False):
        """
        Marks the job as failed, saves the given error message and force exists the process when force_exit=True.
        """
        global last_exit_code

        if not last_exit_code:
            last_exit_code = 1

        with self.git.batch_commit('FAILED'):
            self.set_status('FAILED', add_section=False)
            self.git.commit_json_file('FAIL_MESSAGE', 'aetros/job/crash/error', str(message) if message else '')
            if isinstance(sys.stderr, GeneralLogger):
                self.git.commit_json_file('FAIL_MESSAGE_LAST_LOG', 'aetros/job/crash/last_message', sys.stderr.last_messages)

        self.job_add_status('progress', JOB_STATUS.PROGRESS_STATUS_FAILED)

        self.logger.debug('Crash report stored in commit ' + self.git.git_last_commit)
        self.stop(JOB_STATUS.PROGRESS_STATUS_FAILED, force_exit=force_exit)

    def write_log(self, message):
        """
        Proxy method for GeneralLogger.
        """
        if self.stream_log:
            self.stream_log.write(message)
            return True

    def set_status(self, status, add_section=True):
        """
        Set an arbitrary status, visible in the big wheel of the job view.
        """
        status = str(status)
        if add_section:
            self.section(status)

        self.logger.info('Job status changed to %s ' % (status,))
        self.job_add_status('status', status)

    @property
    def job_id(self):
        return self.git.job_id

    @property
    def config(self):
        return self.job['config']

    def create(self, create_info=None, hyperparameter=None, server='local', insights=False):
        """
        Creates a new job in git and pushes it.

        :param create_info: from the api.create_job_info(id). Contains the config and job info (type, server)
        :param hyperparameter: simple nested dict with key->value, which overwrites stuff from aetros.yml
        :param server: if None, the the job will be assigned to a server.
        :param insights: whether you want to activate insights (for simple models)
        """
        if not create_info:
            create_info = {
                'server': server,
                'config': {
                    'insights': insights,
                    'command': ' '.join(sys.argv)
                }
            }
            config = read_config(self.config_path, logger=self.logger)

            # first transform simple format in the full definition with parameter types
            # (string, number, group, choice_group, etc)
            full_hyperparameters = lose_parameters_to_full(config['parameters'])

            # now extract hyperparameters from full definition, and overwrite stuff using
            # incoming_hyperparameter if available
            hyperparameter = extract_parameters(full_hyperparameters, hyperparameter)
            create_info['config']['parameters'] = hyperparameter

        self.job = create_info

        if 'server' not in self.job and server:
            # setting this disables server assignment
            self.job['server'] = server

        self.job['optimization'] = None
        self.job['type'] = 'custom'

        if 'parameters' not in self.job['config']:
            self.job['config']['parameters'] = {}

        if 'insights' not in self.job['config']:
            self.job['config']['insights'] = insights

        self.git.create_job_id(self.job)

        self.logger.debug("Job created with Git ref " + self.git.ref_head)

        return self.job_id

    def create_task(self, job_id, task_config, name, index):
        # move to git
        task_config['task'] = name
        task_config['index'] = index

        self.git.create_task_id(job_id, task_config)

    def is_simple_model(self):
        if not self.job:
            raise Exception('Job not loaded yet. Use load(id) first.')

        if 'type' in self.job:
            return self.job['type'] == 'simple'

        return False

    def ensure_model_name(self, model_name=None):
        if self.model_name:
            return

        if self.job and 'model' in self.job:
            return self.job['model']

        config = read_config(self.config_path, logger=self.logger)
        self.logger.debug('config: ' + simplejson.dumps(config))

        if model_name is None:
            model_name = os.getenv('AETROS_MODEL_NAME')

        if model_name is None:
            if 'model' not in config or not config['model']:
                sys.stderr.write('Error: No AETROS Trainer model name given. Specify it in aetros.yml `model: user/model-name` or use "aetros init model-name".\n')
                sys.exit(2)

            self.model_name = config['model']
        else:
            self.model_name = model_name

    def get_parameter(self, path, default=None, return_group=False):
        """
        Reads hyperparameter from job configuration. If nothing found use given default.

        :param path: str 
        :param default: *
        :param return_group: If true and path is a choice_group, we return the dict instead of the group name.
        :return: *
        """
        value = read_parameter_by_path(self.job['config']['parameters'], path, return_group)

        if value is None:
            return default

        return value

    def fetch(self, job_id):
        """
        Fetches the job from the server updating the job ref.
        """
        self.git.fetch_job(job_id)

    def load(self, job_id):
        """
        Loads job into index and work-tree, restart its ref and sets as current.

        :param job_id: int
        """
        self.git.read_job(job_id, checkout=self.is_master_process())
        self.load_job_from_ref()

    def load_job_from_ref(self):
        """
        Loads the job.json into self.job
        """
        if not self.job_id:
            raise Exception('Job not loaded yet. Use load(id) first.')

        if not os.path.exists(self.git.work_tree + '/aetros/job.json'):
            raise Exception('Could not load aetros/job.json from git repository. Make sure you have created the job correctly.')

        with open(self.git.work_tree + '/aetros/job.json') as f:
            self.job = simplejson.loads(f.read(), object_pairs_hook=collections.OrderedDict)

        if not self.job:
            raise Exception('Could not parse aetros/job.json from git repository. Make sure you have created the job correctly.')

        self.logger.debug('job: ' + str(self.job))

    def restart(self, job_id):
        self.git.read_job(job_id, checkout=True)

        progress = self.git.contents('aetros/job/status/progress.json')
        if progress is not None:
            progress = float(progress)
        else:
            progress = 0

        if progress >= 2:
            self.logger.error('You can not restart an existing job that was already running. You need to restart the '
                              'job through AETROS Trainer. progress='+str(progress))
            sys.exit(1)

        self.load_job_from_ref()

    def get_job_model(self):
        """
        Returns a new JobModel instance with current loaded job data attached.
        :return: JobModel
        """
        if not self.job:
            raise Exception('Job not loaded yet. Use load(id) first.')

        return JobModel(self.job_id, self.job, self.home_config['storage_dir'])

    def sync_weights(self, push=True):

        if not os.path.exists(self.get_job_model().get_weights_filepath_latest()):
            return

        self.logger.debug("sync weights...")
        self.set_status('SYNC WEIGHTS', add_section=False)

        with open(self.get_job_model().get_weights_filepath_latest(), 'rb') as f:
            import keras.backend
            self.git.commit_file('Added weights', 'aetros/weights/latest.hdf5', f.read())

            image_data_format = None
            if hasattr(keras.backend, 'set_image_data_format'):
                image_data_format = keras.backend.image_data_format()

            info = {
                'framework': 'keras',
                'backend': keras.backend.backend(),
                'image_data_format': image_data_format
            }
            self.git.commit_file('Added weights', 'aetros/weights/latest.json', simplejson.dumps(info))
            if push:
                self.git.push()

        # todo, implement optional saving of self.get_job_model().get_weights_filepath_best()

    def job_add_status(self, key, value):
        self.git.commit_file('STATUS ' + str(value), 'aetros/job/status/' + key + '.json', simplejson.dumps(value, default=invalid_json_values))

    def set_info(self, name, value, commit_end_of_job=False):
        if commit_end_of_job:
            self.git.store_file('aetros/job/info/' + name + '.json', simplejson.dumps(value, default=invalid_json_values))
        else:
            self.git.commit_json_file('INFO ' + name, 'aetros/job/info/' + name, value)

    def set_graph(self, graph):
        self.git.commit_json_file('GRAPH', 'aetros/job/graph', graph)

    def set_system_info(self, key, value, commit_end_of_job=False):
        if commit_end_of_job:
            self.git.store_file('aetros/job/system/' + key + '.json', simplejson.dumps(value, default=invalid_json_values))
        else:
            self.git.commit_json_file('SYSTEM_INFO ' + key, 'aetros/job/system/' + key, value)

    def commit_file(self, path, git_path=None, title=None):
        path = os.path.expanduser(path).strip()

        if not git_path:
            git_path = os.path.relpath(path, os.getcwd())
            git_path = git_path.replace('../', '')
            git_path = git_path.replace('./', '')

        with self.git.batch_commit('FILE ' + (title or git_path)):
            if os.path.isdir(path):
                for file in os.listdir(path):
                    self.commit_file(path + '/' + file)
                return

            if os.path.getsize(path) > 10 * 1024 * 1024 * 1024:
                raise Exception('Can not upload file bigger than 10MB')

            with open(path, 'rb') as f:
                contents = f.read()

            self.git.commit_file('FILE ' + (title or git_path), git_path, contents)

    registered_actions = {}

    def register_action(self,  callback, name=None, label=None, description=None):
        if name is None:
            name = callback.__name__

        args = {}
        inspect_args = inspect.getargspec(callback)
        if inspect_args.args:
            defaults = inspect_args.defaults if inspect_args.defaults else []

            start_default_idx = len(inspect_args.args) - len(defaults)

            for idx, name in enumerate(inspect_args.args):
                args[name] = {'default': None, 'type': 'mixed'}

                if idx >= start_default_idx:
                    default_value = defaults[idx - start_default_idx]
                    arg_type = 'mixed'
                    if isinstance(default_value, six.string_types): arg_type = 'string'
                    if isinstance(default_value, int): arg_type = 'integer'
                    if isinstance(default_value, float): arg_type = 'float'
                    if isinstance(default_value, bool): arg_type = 'bool'

                    args[name] = {'default': default_value, 'type': arg_type}

        value = {
            'label': label,
            'description': description,
            'args': args,
            'instance': self.name
        }

        self.git.store_file('aetros/job/actions/' + name + '/config.json', simplejson.dumps(value, default=invalid_json_values))

        self.registered_actions[name] = value
        value['callback'] = callback

    def on_pause(self):
        pass

    def on_continue(self):
        pass

    def on_action(self, params):
        action_id = params['id']
        action_name = params['name']
        action_value = params['value']

        self.logger.debug("Trigger action: " + str(params))
        self.logger.info("Trigger action %s(%s)" %( action_name, str(action_value)))

        if action_name in ['pause', 'continue']:
            try:
                if action_name == 'pause': return self.on_pause()
                if action_name == 'continue': return self.on_continue()
            except SystemExit:
                raise
            except KeyboardInterrupt:
                raise
            except Exception as e:
                traceback.print_exc()
                self.logger.warning("Trigger action %s failed: %s" % (action_name, type(e).__name__ + ': ' + str(e)))

        if action_name not in self.registered_actions:
            self.logger.warning('Received action ' + str(action_name) + ' but no callback registered.')
            return

        config = self.registered_actions[action_name]
        callback = config['callback']

        action = {
            'name': action_name,
            'value': action_value,
            'time': time.time(),
        }

        self.git.store_file(
            'aetros/job/actions/' + str(action_name) + '/result/' + str(action_id) + '.json',
            simplejson.dumps(action, default=invalid_json_values)
        )

        def done(value):
            result = {
                'value': value,
                'time': time.time()
            }
            # if value is binary, include mimetype and save it in a separate file

            self.git.store_file(
                'aetros/job/actions/' + str(action_name) + '/result/' + str(action_id) + '.json',
                simplejson.dumps(result, default=invalid_json_values)
            )

        kwargs = {}
        try:
            if action_value:
                kwargs = action_value

            if 'done' in config['args']:
                kwargs['done'] = done

            result = callback(**kwargs)
            # returning done as result marks this as async call

            if result is not done:
                # we have no async call
                done(result)

        except SystemExit:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            traceback.print_exc()
            self.logger.warning("Trigger action %s(%s) failed: %s" % (action_name, str(kwargs), type(e).__name__+': '+ str(e)))

            result = {
                'exception': type(e).__name__,
                'message': str(e),
                'time': time.time()
            }

            self.git.store_file(
                'aetros/job/actions/' + str(action_name) + '/result/' + str(action_id) + '.json',
                simplejson.dumps(result, default=invalid_json_values)
            )

    def file_list(self):
        """
        Lists all files in the working directory.
        """
        blacklist = ['.git', 'aetros']
        working_tree = self.git.work_tree

        def recursive(path='.'):
            if os.path.basename(path) in blacklist:
                return 0, 0

            if os.path.isdir(path):
                files = []
                for file in os.listdir(path):
                    if path and path != '.':
                        file = path + '/' + file

                    added_files = recursive(file)
                    files += added_files

                return files
            else:
                if path.endswith('.pyc'):
                    return []

                if is_ignored(path, self.config['ignore']):
                    return []

                return [os.path.relpath(path, working_tree)]

        return recursive(working_tree)

    def add_files(self, working_tree, report=False, message='COMMIT FILES'):
        """
        Commits all files from limited in aetros.yml. `files` is a whitelist, `exclude_files` is a blacklist.
        If both are empty, we commit all files smaller than 10MB.
        :return:
        """
        blacklist = ['.git', 'aetros']

        def add_resursiv(path = '.', report=report):
            self.logger.debug("add_resursiv " + path + ' => ' + str(is_ignored(path, self.config['ignore'])))

            if os.path.basename(path) in blacklist:
                return 0, 0

            if os.path.isdir(path):
                files = 0
                size = 0
                for file in os.listdir(path):
                    if path and path != '.':
                        file = path + '/' + file

                    added_files, added_size = add_resursiv(file)
                    files += added_files
                    size += added_size

                return files, size
            else:
                if path.endswith('.pyc'):
                    return 0, 0

                if is_ignored(path, self.config['ignore']):
                    return 0, 0

                self.logger.debug("added file to job " + path)
                if report:
                    print("Added output job file: " + os.path.relpath(path, working_tree))

                self.git.add_file_path(path, working_tree, verbose=False)

                return 1, os.path.getsize(path)

        return add_resursiv(working_tree, report=report)

    # def job_add_insight(self, x, images, confusion_matrix):
    #     pass

    def job_add_insight(self, x, images, confusion_matrix):
        converted_images = []
        info = {
            '_created': time.time()
        }
        for image in images:
            if not isinstance(image, JobImage):
                raise Exception('job_add_insight only accepts JobImage instances in images argument')

            if '_created' == image.id:
                raise Exception('Insight image id ' + str(image.id) + ' not allowed.')

            converted_images.append({
                'id': image.id,
                'image': self.pil_image_to_jpeg(image.image)
            })
            info[image.id] = {
                'file': image.id+'.jpg',
                'label': image.label,
                'pos': image.pos
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

    def collect_environment(self, overwrite_variables={}):
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
        env['variables'].update(overwrite_variables)

        if 'AETROS_SSH_KEY' in env['variables']: del env['variables']['AETROS_SSH_KEY']
        if 'AETROS_SSH_KEY_BASE64' in env['variables']: del env['variables']['AETROS_SSH_KEY_BASE64']

        env['pip_packages'] = sorted([[i.key, i.version] for i in pip.get_installed_distributions()])
        self.set_system_info('environment', env)

    def collect_device_information(self, gpu_ids):
        import aetros.cuda_gpu

        try:
            if gpu_ids:
                self.set_system_info('cuda_version', aetros.cuda_gpu.get_version())
                gpus = {}
                for gpu_id, gpu in enumerate(aetros.cuda_gpu.get_ordered_devices()):
                    if gpu_id in gpu_ids:
                        gpus[gpu_id] = gpu

                self.set_system_info('gpus', gpus)
        except CudaNotImplementedException:
            self.logger.warning("Could not collect GPU/CUDA system information.")

        if self.get_job_model().has_dpu():
            self.set_system_info('dpus', [{'memory': 64*1024*1024*1024}])

    def collect_system_information(self):
        import psutil

        mem = psutil.virtual_memory()

        with self.git.batch_commit('JOB_SYSTEM_INFORMATION'):
            self.set_system_info('memory_total', mem.total)

            import cpuinfo
            cpu = cpuinfo.get_cpu_info()
            self.set_system_info('cpu_name', cpu['brand'])
            self.set_system_info('cpu', [cpu['hz_actual_raw'][0], cpu['count']])

    stdout_api_channels = {}

    def handle_stdout_api(self, data):
        action = data['aetros']
        del data['aetros']

        def validate_action(requires_attributes):
            for attr in requires_attributes:
                if attr not in data:
                    self.logger.warning("AETROS stdout API call %s requires value for '%s'. Following ignored: %s" % (action, attr, str(data)))
                    return False

            return True

        def failed(message):
            self.logger.warning(
                "AETROS stdout API call %s failed: %s Following ignored: %s" % (action, message, str(data)))

        def default(attr, default=None):
            return data[attr] if attr in data else default

        if action == 'epoch' or action == 'progress':
            if 'epoch' not in data and 'total' not in data:
                failed("requires value for 'epoch' or 'total'")
                return False

            self.progress(**data)
            return True

        if action == 'batch':
            if validate_action(['batch', 'total', 'batch_size']):
                self.batch(**data)
                return True

        if action == 'step':
            if validate_action(['step', 'total']):
                self.step(**data)
                return True

        if action == 'sample':
            if validate_action(['sample', 'total']):
                self.sample(**data)
                return True

        if action == 'info':
            if validate_action(['name', 'value']):
                self.set_info(**data)
                return True

        if action == 'status':
            if validate_action(['status']):
                self.set_status(**data)
                return True

        if action == 'speed':
            if validate_action(['x', 'speed']):
                self.report_speed(**data)
                return True

        if action == 'create-channel':
            if validate_action(['name']):
                if data['name'] in self.stdout_api_channels:
                    failed("Channel %s already defined. " % (data['name'], ))
                else:
                    self.stdout_api_channels[data['name']] = self.create_channel(**data)
                    return True

        if action == 'channel':
            if validate_action(['name', 'x', 'y']):
                if data['name'] not in self.stdout_api_channels:
                    self.stdout_api_channels[data['name']] = self.create_channel(data['name'])

                self.stdout_api_channels[data['name']].send(data['x'], data['y'])
                return True

        # if action == 'insight':
        #     if validate_action(['x']):
        #
        #         self.job_add_insight(data['x'])
        #         return True

        if action == 'abort':
            self.abort()

        if action == 'fail':
            self.fail(default('message'))

        return False
