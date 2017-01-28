from __future__ import print_function, division

from __future__ import absolute_import
import io
import logging
import os
import pprint
import re
import signal
import subprocess
import sys
import traceback

from . import keras_model_utils
from .backend import JobBackend
from .logger import GeneralLogger
from .JobModel import JobModel
from .MonitorThread import MonitoringThread
from .Trainer import Trainer
from .keras_model_utils import ensure_dir
import six


def start(job_id, hyperparameter=None, dataset_id=None, server_id='local', insights=False, insights_sample_path=None, api_token=None):
    """
    Starts the training process with all logging of a job_id
    """

    job_backend = JobBackend(api_token=api_token)

    if '/' in job_id:
        print("Create job ...")
        job_id = job_backend.create(job_id, server_id=server_id, hyperparameter=hyperparameter, dataset_id=dataset_id, insights=insights)
        job_backend.load(job_id)

        print("Job %s#%d (%s) created and started. Open http://%s/trainer/app#/training=%s to monitor the training." %
              (job_backend.model_id, job_backend.job_index, job_backend.job_id, job_backend.host, job_id))
    else:
        job_backend.load(job_id)

    if not len(job_backend.get_job_model().config):
        raise Exception('Job does not have a configuration. Make sure you created the job via AETROS TRAINER')

    if job_backend.is_keras_model():
        start_keras(job_backend, insights_sample_path)
    else:
        start_custom(job_backend)


def start_custom(job_backend):
    job_model = job_backend.get_job_model()
    settings = job_model.config['settings']

    if 'git' not in settings or not settings['git']:
        raise Exception('Git url is not configured. Aborted')

    if 'git_python_script' not in settings or not settings['git_python_script']:
        raise Exception('Git python script is not configured. Aborted')

    git_python_script = settings['git_python_script']
    git_branch = 'master'
    git_url = settings['git']
    if 'git_branch' in settings and settings['git_branch']:
        git_branch = settings['git_branch']

    root = './aetros-job/'
    if not os.path.exists(root):
        os.mkdir(root)

    my_env = os.environ.copy()
    if 'PYTHONPATH' not in my_env:
        my_env['PYTHONPATH'] = ''
    my_env['PYTHONPATH'] += ':' + os.getcwd()
    my_env['AETROS_JOB_ID'] = job_model.id

    os.chdir(root)

    print("Setup git repository %s in %s" % (git_url, root + job_model.model_id))

    try:
        if not os.path.exists(job_model.model_id):
            args = ['git', 'clone', git_url, job_model.model_id]
            code = subprocess.call(args, stdout=sys.stdout, stderr=sys.stderr)
            if code != 0:
                raise Exception('Could not clone repository %s to %s' %(git_url, job_model.model_id))
        else:
            # repository seems to exists already, make hard reset and git pull origin
            # check if requested branch is loaded
            os.chdir(job_model.model_id)
            branches = '\n' + subprocess.check_output(['git', 'branch']) + '\n'
            m = re.search('\* ([^\s]+)', branches)
            current_branch = m.group(1) if m else None

            if current_branch == git_branch:
                print("Reset all local changes git repo")
                subprocess.call(['git', 'reset', '--hard'])
                subprocess.call(['git', 'clean', '-fd'])

                print("Update local git repo: git pull origin " + git_branch)
                code = subprocess.call(['git', 'pull', 'origin', git_branch])

                if code != 0:
                    raise Exception('Could not "git pull origin %s" repository %s to %s' %(git_branch, git_url, job_model.model_id))

            if current_branch != git_branch:
                branch_checked_out = '\n ' + git_branch + '\n' in branches
                if not branch_checked_out:
                    subprocess.call(['git', 'fetch', 'origin', git_branch + ':' + git_branch])

                subprocess.call(['git', 'checkout', git_branch])

                print("Update local git repo: git pull origin " + git_branch)
                subprocess.call(['git', 'pull', 'origin', git_branch])

    except subprocess.CalledProcessError as e:
        raise Exception('Could not run "%s" for repository %s in %s' %(e.cmd, git_url, job_model.model_id))

    print("\nExecuting %s" %(git_python_script,))
    args = [sys.executable, git_python_script]
    subprocess.Popen(args, close_fds=True, env=my_env).wait()


def start_keras(job_backend, insights_sample_path=None):
    job_model = job_backend.get_job_model()

    # we need to import keras here, so we know which backend is used (and whether GPU is used)
    from keras import backend as K
    # all our shapes are theano schema. (channels, height, width)
    K.set_image_dim_ordering('th')

    job_backend.start()

    ensure_dir('aetros-cli-data/models/%s/%s' % (job_model.model_id, job_model.id))

    log = io.open('aetros-cli-data/models/%s/%s/output.log' % (job_model.model_id, job_model.id), 'w', encoding='utf8')
    log.truncate()

    general_logger = GeneralLogger(log, job_backend)

    from .KerasLogger import KerasLogger
    trainer = Trainer(job_backend, general_logger)
    keras_logger = KerasLogger(trainer, job_backend, general_logger)
    keras_logger.insights_sample_path = insights_sample_path
    trainer.callbacks.append(keras_logger)

    sys.stdout = general_logger
    sys.stderr = general_logger

    def ctrlc(sig, frame):
        print("signal %s received\n" % id)
        raise KeyboardInterrupt("CTRL-C!")

    signal.signal(signal.SIGINT, ctrlc)

    try:
        print("Setup job")
        keras_model_utils.job_prepare(job_model)
        job_backend.progress(0, job_backend.job['config']['settings']['epochs'])

        print("Start job")
        keras_model_utils.job_start(job_model, trainer, keras_logger, general_logger)

        job_backend.sync_weights()
        job_backend.done()

        print("Done.")
        sys.exit(0)
    except KeyboardInterrupt:
        trainer.set_status('ABORTED')
        print('Early stopping ...')

        if job_backend.stop_requested:
            print(' ... stop requested through trainer.')

        if trainer.model:
            trainer.model.stop_training = True

        job_backend.sync_weights()
        job_backend.abort()
        print("Aborted.")
        sys.exit(1)

    except Exception as e:
        print("Crashed ...")

        if trainer.model:
            trainer.model.stop_training = True

        log.write(six.text_type(traceback.format_exc()))
        logging.error(traceback.format_exc())

        job_backend.crash(e)
        print("Crashed.")
        raise e
