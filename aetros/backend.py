from __future__ import absolute_import
from __future__ import print_function

import atexit
import inspect
import os
import traceback

from threading import Lock

import collections

import math
import signal
import time

import numpy as np
import simplejson
import six
import PIL.Image
import sys

from aetros.JobModel import JobModel
from aetros.client import JobClient
from aetros.const import JOB_STATUS
from aetros.cuda_gpu import CudaNotImplementedException
from aetros.git import Git
from aetros.logger import GeneralLogger
from aetros.utils import git, invalid_json_values, read_config, is_ignored, prepend_signal_handler, raise_sigint, \
    read_parameter_by_path, stop_time, read_home_config, lose_parameters_to_full, extract_parameters, get_logger, \
    is_debug, find_config
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


class StdoutApiException(Exception): pass


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


def context():
    """
    Returns a new JobBackend instance which connects to AETROS Trainer
    based on "model" in aetros.yml or (internal: env:AETROS_MODEL_NAME environment variable).

    internal: If env:AETROS_JOB_ID is not defined, it creates a new job.

    Job is ended either by calling JobBackend.done(), JobBackend.fail() or JobBackend.abort().
    If the script ends without calling one of the methods above, JobBackend.stop() is called and exit code defines
    whether it is a fail() or done() result.

    :return: JobBackend
    """
    job = JobBackend()

    offline = False
    if '1' == os.getenv('AETROS_OFFLINE', ''):
        offline = True

    if os.getenv('AETROS_JOB_ID'):
        job.load(os.getenv('AETROS_JOB_ID'))
        if not offline:
            job.connect()
    else:
        job.create()
        if not offline:
            job.connect()

    job.start(offline=offline)

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
        self.stream.write('"time", "x","training","validation"\n')

    def send(self, x, training, validation):
        line = simplejson.dumps([self.job_backend.get_run_time(), x, training, validation])[1:-1]

        self.lock.acquire()
        try:
            self.stream.write(line + "\n")
            self.job_backend.git.store_file('aetros/job/channel/' + self.name + '/last.csv', line)
        finally:
            self.lock.release()


class JobImage:
    def __init__(self, name, pil_image, label=None, pos=None):
        self.id = name
        if not isinstance(pil_image, PIL.Image.Image):
            raise Exception('JobImage requires a PIL.Image as image argument.')

        self.image = pil_image
        self.label = label
        self.pos = pos

        if self.pos is None:
            self.pos = time.time()


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
            'timed': True,
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

        line = simplejson.dumps(['time', 'x'] + [str(x['name']) for x in traces])[1:-1]
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

        line = simplejson.dumps([self.job_backend.get_run_time(), x] + y)[1:-1]

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

    def __init__(self, model_name=None, logger=None, config_path='aetros.yml', name=None):
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
        self.start_time = time.time()

        # whether we are in paused state
        self.is_paused = False

        # whether on_shutdown has been called and thus the python interpreter is dying.
        self.in_shutdown = False

        self.insight_images_info = {}
        self.insight_created = []

        self.monitoring_thread = None

        if not self.logger:
            self.logger = get_logger(os.getenv('AETROS_JOB_NAME', 'aetros-job'))

        self.last_progress_call = None
        self.job_ids = []
        self.in_request = False
        self.stop_requested = False
        self.stop_requested_force = False

        self.kpi_channel = None
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
        self.git = Git(self.logger, self.client, self.home_config, self.model_name, self.is_master_process())

        self.logger.debug("Started tracking of job files in git %s for remote %s" % (self.git.git_path, self.git.origin_url))

    @property
    def log_level(self):
        if os.getenv('DEBUG') == '1':
            return 'DEBUG'

        return 'INFO'

    @property
    def host(self):
        return self.home_config['host']

    def get_run_time(self, precision=3):
        return round(time.time() - self.start_time, precision)

    def section(self, title):
        title = title.replace("\t", "  ")
        seconds = self.get_run_time()
        line = "## %s\t%.2f\n" % (title, seconds)
        sys.stdout.write(line)
        sys.stdout.flush()

    def on_registration_failed(self, params):
        self.logger.warning("Connecting to AETROS Trainer at %s failed. Reasons: %s" % (self.host, params['reason'],))
        if 'Permission denied' in params['reason']:
            self.logger.warning("Make sure you have saved your ssh pub key in your AETROS Trainer user account.")

    def on_client_offline(self, params):
        if self.is_master_process():
            self.logger.warning("Could not establish a connection. We stopped automatic syncing.")
            self.logger.warning("You can publish later this job to AETROS Trainer by executing following command.")
            self.logger.warning("$ aetros job-push " + self.job_id[0:9])

        self.git.online = False

    def on_registration(self, params):
        pass

    def on_signusr1(self, signal, frame):
        self.logger.warning("USR1: backend job_id=%s (running=%s, ended=%s), client (online=%s, active=%s, registered=%s, "
                            "connected=%s, queue=%s), git (active_thread=%s, last_push_time=%s)." % (
          str(self.job_id),
          str(self.running),
          str(self.ended),
          str(self.client.online),
          str(self.client.active),
          str(self.client.registered),
          str(self.client.connected),
          str([str(i)+':'+str(len(x)) for i, x in six.iteritems(self.client.queues)]),
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
            self.logger.warning('Received signal '+str(sig)+'. Send again to force stop. Stopping ...')
        else:
            self.logger.debug("Got child signal " + str(sig))

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

    def batch(self, batch, total, size, label='BATCH', speed_label='SAMPLES/S'):
        self.step(batch, total, size=size, label=label, speed_label=speed_label)

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
            x = round(time.time()-self.start_time, 3)

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

    def create_progress(self, name, total_steps=100):
        if name in self.progresses:
            return self.progresses[name]

        class Controller():
            def __init__(self, git, name, total_steps=100):
                self.started = False
                self.stopped = False
                self.lock = Lock()
                self.name = name
                self.step = 0
                self.steps = total_steps
                self.eta = 0
                self.last_call = 0
                self.git = git
                self._label = name

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
                self.git.store_file('aetros/job/progress/' + self.name + '.json', simplejson.dumps(info))

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

        self.progresses[name] = Controller(self.git, name, total_steps)

        return self.progresses[name]

    def epoch(self, epoch=None, total=None):
        self.progress(epoch, total)

    def progress(self, progress=None, total=None):
        self.current_epoch = self.current_epoch if progress is None else progress
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

    def create_loss_channel(self, name='loss', xaxis=None, yaxis=None, layout=None):
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

    def connect(self):
        self.client.configure(self.model_name, self.job_id, self.name)
        return self.client.start(['', 'files'])

    def start(self, collect_system=True, offline=False, push=True):
        if self.started:
            raise Exception('Job was already started.')

        if self.running:
            raise Exception('Job already running.')

        if not self.job_id:
            raise Exception('No job id found. Use create() first.')

        if not self.job:
            raise Exception('Job not loaded')

        prepend_signal_handler(signal.SIGINT, self.on_sigint)
        self.start_time = time.time()

        self.started = True
        self.running = True
        self.ended = False

        on_shutdown.started_jobs.append(self)

        self.client.configure(self.model_name, self.job_id, self.name)

        if not offline:
            # Marks client as active if not already. If not already starts to connect to the server
            self.client.start(['', 'files'])
        else:
            self.logger.debug('Job backend not started since offline.')

        if self.is_master_process():
            # this is the process that actually starts the job.
            # other sub-processes may only modify other data.
            self.git.commit_file('JOB_STARTED', 'aetros/job/times/started.json', simplejson.dumps(self.start_time))
            self.job_add_status('progress', JOB_STATUS.PROGRESS_STATUS_STARTED)
            self.git.store_file('aetros/job/times/elapsed.json', str(0))

            if collect_system:
                self.collect_system_information()
                self.collect_environment()

            # make sure we get the progress first, before monitoring sends elapses and
            # updates the job cache
            if not offline and self.client.online and push:
                self.git.push()
                self.git.start_push_sync()

            if collect_system:
                self.start_monitoring()

            # log stdout to Git by using self.write_log -> git:stream_file
            self.stream_log = self.git.stream_file('aetros/job/log.txt', fast_lane=False)
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

    def start_monitoring(self, cpu_cores=1, gpu_devices=None, docker_container=None):
        if not self.monitoring_thread:
            self.monitoring_thread = MonitoringThread(self, cpu_cores, gpu_devices, docker_container)
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

        self.logger.debug('on_shutdown, stopped=%s, ended=%s, early_stop=%s, stop_requested=%s'
                          % (str(self.stopped), str(self.ended), str(self.in_early_stop), str(self.stop_requested)))
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
                    self.stop(force_exit=True)
                else:
                    # master process
                    self.fail('Force stopped.', force_exit=True)
            else:
                if not self.is_master_process():
                    # if not master process, we just stop everything. status/progress is set by master
                    self.stop()
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

        self.stop(JOB_STATUS.PROGRESS_STATUS_DONE, force_exit=force_exit)

    def send_std_buffer(self):
        if isinstance(sys.stdout, GeneralLogger):
            sys.stdout.send_buffer()

        if isinstance(sys.stderr, GeneralLogger):
            sys.stderr.send_buffer()

    def stop(self, progress=None, force_exit=False):
        global last_exit_code

        if self.stopped:
            return

        if self.is_master_process():
            self.section('Ended')

        self.logger.debug("stop: " + str(progress))

        self.send_std_buffer()

        self.stopped = True
        self.ended = True
        self.running = False

        if self.is_master_process() and progress is not None:
            # if not master process, the master process will set it
            self.job_add_status('progress', progress)

        exit_code = last_exit_code or 0
        if progress == JOB_STATUS.PROGRESS_STATUS_DONE:
            exit_code = 0

        if progress == JOB_STATUS.PROGRESS_STATUS_ABORTED:
            exit_code = 1

        if progress == JOB_STATUS.PROGRESS_STATUS_FAILED:
            exit_code = 2

        if self.is_master_process():
            self.set_system_info('exit_code', exit_code)

        # stop push thread and commit STREAMED/STORE END files in local git
        self.logger.debug("Git stopping ...")
        self.git.stop()

        if self.client.online and not force_exit:
            # make sure all queues are empty and everything has been sent
            self.logger.debug("Wait for queue empty and store Git blobs on server: master=" +str(self.is_master_process()))

            # Say server to store received files as blob in remote Git
            self.client.send({'type': 'sync-blob'}, channel='')
            self.client.send({'type': 'sync-blob'}, channel='files')

            report = self.is_master_process() or is_debug()

            if report:
                sys.stdout.write("Uploading last job data ... ")

            self.client.wait_until_queue_empty(['', 'files'], report=report, clear_end=True)
            self.logger.debug("Blobs remotely stored, build latest git pack now")
            # all further client.send calls won't be included in the final git push calculation
            # and might be sent again.

            failure_in_last_sync = False

            # do now the last final git push, where we upload commits and trees.
            # blobs should be added already via streaming
            if self.is_master_process():
                # non-master commit and upload only.
                # master tracks what commits habe been sent already
                if self.git.push() is False:
                    failure_in_last_sync = True

            # send all last messages and git pack
            self.logger.debug("Last wait_until_queue_empty")
            self.client.wait_until_queue_empty(['', 'files'], report=report, clear_end=False)

            # it's important to have it here, since its tracks not only hardware but also network speed
            # for uploading last messages and Git.
            # Also, after each message we get from this thread on the server, we check if the job
            # should be ended/terminated or not.
            if self.monitoring_thread:
                self.monitoring_thread.stop()
                self.monitoring_thread.join()

            # send last monitoring stuff and close channels
            self.client.wait_sending_last_messages()

            if self.is_master_process():
                sys.stdout.write(" done.\n")

            # wait for end of client. Server will now close connection when ready.
            self.client.end()

            if self.is_master_process():
                # check if we have uncommitted stuff
                objects_to_sync, types = self.git.diff_objects(self.git.get_head_commit())

                if objects_to_sync:
                    failure_in_last_sync = True

            if self.is_master_process() and failure_in_last_sync:
                self.logger.warning("Not all job data have been uploaded.")
                self.logger.warning("Please run following command to make sure your job is stored on the server.")
                self.logger.warning("$ aetros job-push " + self.job_id[0:9])

        elif self.is_master_process():
            self.logger.warning("Not all job data have been uploaded because you went offline.")
            self.logger.warning("Run following command to make sure your job is stored on the server.")
            self.logger.warning("$ aetros job-push " + self.job_id[0:9])

        if self.is_master_process():
            # remove the index file
            # non-master use the same as master, so master cleans up
            self.git.clean_up()

        # make sure client is really stopped
        self.client.close()

        self.logger.debug("Stopped %s with last commit %s." % (self.git.ref_head, self.git.get_head_commit()))

        if force_exit:
            self.on_force_exit()
            os._exit(exit_code)
        elif not self.in_shutdown:
            sys.exit(exit_code)

    def abort(self, force_exit=False):
        if not self.running:
            return

        self.set_status('ABORTED', add_section=False)

        self.stop(JOB_STATUS.PROGRESS_STATUS_ABORTED, force_exit=force_exit)

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

        self.logger.debug('Crash report stored in commit ' + self.git.get_head_commit())
        self.stop(JOB_STATUS.PROGRESS_STATUS_FAILED, force_exit=force_exit)

    def write_log(self, message):
        """
        Proxy method for GeneralLogger.
        """
        if self.stream_log and not self.ended:
            # points to the Git stream write
            self.stream_log.write(message)
            return True

    def set_status(self, status, add_section=True):
        """
        Set an arbitrary status, visible in the big wheel of the job view.
        """
        status = str(status)

        if add_section:
            self.section(status)

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
            config = find_config(self.config_path, logger=self.logger)

            if not config['model']:
                raise Exception('AETROS config file (aetros.yml) not found.')

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

        config = find_config(self.config_path, logger=self.logger)
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
        path = 'aetros/job/status/' + key + '.json'
        data = simplejson.dumps(value, default=invalid_json_values)
        self.git.commit_file('STATUS ' + str(value), path, data)

        if self.client.online is not False:
            # just so have it faster
            self.client.send({'type': 'store-blob', 'path': path, 'data': data}, channel='')

    def set_info(self, name, value, commit_end_of_job=False):
        if commit_end_of_job:
            self.git.store_file('aetros/job/info/' + name + '.json',
                                simplejson.dumps(value, default=invalid_json_values))
        else:
            self.git.commit_json_file('INFO ' + name, 'aetros/job/info/' + name, value)

    def set_graph(self, graph):
        self.git.commit_json_file('GRAPH', 'aetros/job/graph', graph)

    def set_system_info(self, key, value, commit_end_of_job=False):
        if commit_end_of_job:
            self.git.store_file('aetros/job/system/' + key + '.json',
                                simplejson.dumps(value, default=invalid_json_values))
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

            if os.path.getsize(path) > 10 * 1024 * 1024:
                self.logger.error('Can not upload files bigger than 10MB: ' + str(path))
                return

            with open(path, 'rb') as f:
                contents = f.read()

            self.git.commit_file('FILE ' + (title or git_path), git_path, contents)

    registered_actions = {}

    def register_action(self, callback, name=None, label=None, description=None):
        if name is None:
            name = callback.__name__

        args = {}
        inspect_args = inspect.getargspec(callback)
        if inspect_args.args:
            defaults = inspect_args.defaults if inspect_args.defaults else []

            start_default_idx = len(inspect_args.args) - len(defaults)

            for idx, argname in enumerate(inspect_args.args):
                args[argname] = {'default': None, 'type': 'mixed'}

                if idx >= start_default_idx:
                    default_value = defaults[idx - start_default_idx]
                    arg_type = 'mixed'
                    if isinstance(default_value, six.string_types): arg_type = 'string'
                    if isinstance(default_value, int): arg_type = 'integer'
                    if isinstance(default_value, float): arg_type = 'float'
                    if isinstance(default_value, bool): arg_type = 'bool'

                    args[argname] = {'default': default_value, 'type': arg_type}

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
            # Received action ' + str(action_name) + ' but no callback registered.')
            return

        self.logger.debug("Trigger action: " + str(params))
        self.logger.info("Trigger action %s(%s)" %( action_name, str(action_value)))

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

    def add_files(self, working_tree, report=False):
        """
        Commits all files from limited in aetros.yml. `files` is a whitelist, `exclude_files` is a blacklist.
        If both are empty, we commit all files smaller than 10MB.
        :return:
        """
        blacklist = ['.git']

        def add_resursiv(path = '.', report=report):
            if os.path.basename(path) in blacklist:
                return 0, 0

            if working_tree + '/aetros' == path:
                # ignore in work_tree the folder ./aetros/, as it could be
                # that we checked out a job and start it again.
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
                    print("Added job file: " + os.path.relpath(path, working_tree))

                self.git.add_file_path(path, working_tree, verbose=False)

                return 1, os.path.getsize(path)

        return add_resursiv(working_tree, report=report)

    def add_embedding_word2vec(self, x, path, dimensions=None, header_with_dimensions=True):
        """
        Parse the word2vec file and extracts vectors as bytes and labels as TSV file.
        The format is simple: It a UTF-8 encoded file, each word + vectors separated by new line.
        Vector is space separated.
        At the very first line might be dimensions, given as space separated value.


        Line 1: 2 4\n
        Line 2: word 200.3 4004.4 34.2 22.3\n
        Line 3: word2 20.0 4.4 4.2 0.022\n
        and so on

        For performance reasons, you should prefer add_embedding_path().

        """
        if path.endswith('.txt'):
            if not os.path.exists(path):
                raise Exception("Given word2vec file does not exist: " + path)

            f = open(path, 'r')

            if not header_with_dimensions and not dimensions:
                raise Exception('Either the word2vec file should contain the dimensions as header or it needs to be'
                                'specified manually using dimensions=[x,y] argument.')

            if header_with_dimensions:
                line = f.readline()
                if ' ' not in line:
                    raise Exception('Given word2vec file should have in first line the dimensions, e.g.: 1000 200')
                dimensions = np.fromstring(line, dtype=np.uint, sep=' ').tolist()

            vectors = b''
            labels = ''

            line_pos = 1 if header_with_dimensions else 0

            # while '' != (line = f.readline())
            for line in iter(f.readline, ''):
                line_pos += 1
                space_pos = line.find(' ')
                if -1 == space_pos:
                    message = 'Given word2vec does not have correct format in line ' + str(line_pos)
                    message += '\nGot: ' + str(line)
                    raise Exception(message)

                labels += line[:space_pos] + '\n'
                vectors += np.fromstring(line[space_pos+1:], dtype=np.float32, sep=' ').tobytes()
        else:
            raise Exception("Given word2vec is not a .txt file. Other file formats are not supported.")

        info = {
            'dimensions': dimensions
        }

        name = os.path.basename(path)
        self._ensure_insight(x)
        remote_path = 'aetros/job/insight/'+str(x)+'/embedding/'
        with self.git.batch_commit('INSIGHT_EMBEDDING ' + str(x)):
            self.git.commit_file('WORD2VEC', remote_path + name + '/tensor.bytes', vectors)
            self.git.commit_file('WORD2VEC', remote_path + name + '/metadata.tsv', labels)
            self.git.commit_file('WORD2VEC', remote_path + name + '/info.json', simplejson.dumps(info))

    def add_embedding_path(self, x, dimensions, tensor, metadata=None, image_shape=None, image=None):
        """
        Adds a new embedding with optional metadata.

        Vectors is a floats64 bytes file, no separators sum(dimensions)*floats64 long.

        Metadata is a TSV file. If only one column long (=no tab separator per line), then there's no need for a header.
        If you have more than one column, use the first line as header.

        Metadata example:

        Label\tcount\n
        red\t4\n
        yellow\t6\n

        """
        if not os.path.exists(tensor):
            raise Exception("Given embedding vectors file does not exist: " + tensor)

        if metadata and not os.path.exists(metadata):
            raise Exception("Given embedding metadata file does not exist: " + metadata)

        name = os.path.basename(tensor)
        self._ensure_insight(x)
        remote_path = 'aetros/job/insight/'+str(x)+'/embedding/'

        info = {
            'dimensions': dimensions,
            'image_shape': image_shape,
            'image': os.path.basename(image) if image else None,
        }

        with self.git.lock_write():
            self.git.add_file_path(remote_path + name + '/tensor.bytes', tensor)
            self.git.add_file_path(remote_path + name + '/metadata.tsv', metadata)
            self.git.add_file(remote_path + name + '/info.json', simplejson.dumps(info))

            if image:
                self.git.add_file(remote_path + name + '/' + os.path.basename(image), image)

            self.git.commit_index('INSIGHT_EMBEDDING ' + str(x))

    def add_insight_image_path(self, x, path, name=None, label=None):
        image = PIL.Image.open(path)

        if not name:
            name = os.path.basename(path)

        return self.add_insight_image(x, JobImage(name, image, label))

    def add_insight_image(self, x, image):
        self.add_insight_images(x, [image])

    def add_insight_images(self, x, images):
        converted_images = []

        self._ensure_insight(x)

        if x not in self.insight_images_info:
            self.insight_images_info[x] = {}

        for image in images:
            if not isinstance(image, JobImage):
                raise Exception('job_add_insight only accepts JobImage instances in images argument')

            if image.id in self.insight_images_info[x]:
                continue

            converted_images.append({
                'id': image.id,
                'image': self.pil_image_to_jpeg(image.image)
            })

            self.insight_images_info[x][image.id] = {
                'file': image.id+'.jpg',
                'label': image.label,
                'pos': image.pos
            }

        with self.git.batch_commit('INSIGHT_IMAGES ' + str(x)):
            for image in converted_images:
                remote_path = 'aetros/job/insight/'+str(x)+'/image/'+image['id']+'.jpg'
                self.git.commit_file('IMAGE ' + str(image['id']), remote_path, image['image'])

            remote_path = 'aetros/job/insight/' + str(x) + '/info.json'
            self.git.commit_file('IMAGE INFO', remote_path, simplejson.dumps(self.insight_images_info[x]))

    def add_insight_confusion_matrix(self, x, confusion_matrix):
        self._ensure_insight(x)
        remote_path = 'aetros/job/insight/' + str(x) + '/confusion_matrix.json'
        self.git.commit_file('INSIGHT CONFUSION_MATRIX ' + str(x), remote_path, simplejson.dumps(confusion_matrix))

    def job_add_insight(self, x, images=None, confusion_matrix=None):
        if images:
            self.add_insight_images(x, images)

        if confusion_matrix:
            self.add_insight_confusion_matrix(x, confusion_matrix)

    def _ensure_insight(self, x):
        if x in self.insight_created: return

        self.insight_created.append(x)
        remote_path = 'aetros/job/insight/' + str(x) + '/created'
        self.git.commit_file('WORD2VEC ' + str(x), remote_path, str(time.time()))

    def pil_image_to_jpeg(self, image):
        buffer = six.BytesIO()

        image.save(buffer, format="JPEG", optimize=True, quality=70)
        return buffer.getvalue()

    def collect_environment(self, overwrite_variables=None):
        import socket
        import os
        import pip
        import platform

        env = {}

        if not overwrite_variables:
            overwrite_variables = {}

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
                    raise StdoutApiException("AETROS stdout API call %s requires value for '%s'. " % (action, attr))

            return True

        def failed(message):
            raise StdoutApiException(
                "AETROS stdout API call %s failed: %s Following ignored: %s" % (action, message, str(data)))

        def default(attr, default=None):
            return data[attr] if attr in data else default

        if action == 'progress':
            self.progress(**data)
            return True

        if action == 'epoch':
            self.epoch(**data)
            return True

        if action == 'batch':
            if validate_action(['batch', 'total', 'size']):
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

        if action == 'add_embedding_word2vec':
            if validate_action(['x', 'path']):
                self.add_embedding_word2vec(**data)
                return True

        if action == 'add_insight_image':
            if validate_action(['x', 'path']):
                self.add_insight_image_path(**data)
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

        if action == 'loss':
            if validate_action(['x', 'training', 'validation']):
                if 'loss' not in self.stdout_api_channels:
                    self.stdout_api_channels['loss'] = self.create_loss_channel('loss')

                self.stdout_api_channels['loss'].send(data['x'], data['training'], data['validation'])
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
