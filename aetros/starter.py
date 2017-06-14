from __future__ import print_function, division

from __future__ import absolute_import
import io
import logging
import os
import pprint
import re
import shutil
import signal
import subprocess
import sys
import traceback

from aetros.utils import git
from . import keras_model_utils
from .backend import JobBackend
from .Trainer import Trainer
from .keras_model_utils import ensure_dir
import six


def start(id, hyperparameter=None, dataset_id=None, server_id='local', insights=False, api_key=None):
    """
    Starts the training process with all logging of a job_id
    :type id: string : job id or model name
    """

    job_backend = JobBackend(api_key=api_key)
    job_backend.ensure_job(id, hyperparameter, dataset_id, server_id, insights)
    job_backend.setup_std_output_logging()
    job_backend.start()

    if not len(job_backend.get_job_model().config):
        raise Exception('Job does not have a configuration. Make sure you created the job via AETROS TRAINER')

    if api_key:
        os.environ["API_KEY"] = api_key

    if job_backend.is_keras_model():
        start_keras(job_backend)
    else:
        start_custom(job_backend)


def start_custom(job_backend):
    job_model = job_backend.get_job_model()
    job_config = job_model.config
    model_settings = job_model.model_settings

    if 'git' not in model_settings or not model_settings['git']:
        raise Exception('Git url is not configured. Aborted')

    if 'gitPythonScript' not in model_settings or not model_settings['gitPythonScript']:
        raise Exception('Git python script is not configured. Aborted')

    git_python_script = model_settings['gitPythonScript']
    git_branch = 'master'
    git_url = model_settings['git']
    git_commit = None


    if 'gitBranch' in job_config and job_config['gitBranch']:
        git_branch = job_config['gitBranch']
    elif 'gitBranch' in model_settings and model_settings['gitBranch']:
        git_branch = model_settings['gitBranch']

    if 'gitCommit' in job_config and job_config['gitCommit']:
        git_commit = job_config['gitCommit']
    elif 'gitCommit' in model_settings and model_settings['gitCommit']:
        git_commit = model_settings['gitCommit']

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
            code = subprocess_call(args)
            if code != 0:
                raise Exception('Could not clone repository %s to %s' %(git_url, job_model.model_id))
            os.chdir(job_model.model_id)
        else:
            # repository seems to exists already, make hard reset and git pull origin
            # check if requested branch is loaded
            os.chdir(job_model.model_id)

            # check if remote origin is current git_url. if not remove origin and re-add
            # todo

        # make sure the requested branch is existent in local git. Target FETCH_HEAD to this branch.
        subprocess_call(['git', 'fetch', 'origin', git_branch])

        current_branch = git.get_current_branch()
        # make sure the requested branch is checked out
        if current_branch != git_branch:
            subprocess_call(['git', 'checkout', git_branch])


        if git_commit and git_commit != git.get_current_commit_hash():
            print("Checkout commit %s" % [git_commit])
            subprocess_call(['git', 'reset', '--hard', git_commit]) #reset to requested commit of FETCH_HEAD, revert all changes to repo files
        else:
            print("Checkout newest commit")
            # checkout newest commit of FETCH_HEAD, revert all changes to repo files
            subprocess_call(['git', 'reset', '--hard', 'FETCH_HEAD'])

        # make this configurable, since cache should not always be cleared on the same server.
        # clean created files that do not belong to the repo
        # print("Delete all files that do not belong to the git repository.")
        # subprocess_call(['git', 'clean', '-fd'])

    except subprocess.CalledProcessError as e:
        raise Exception('Could not run "%s" for repository %s in %s. Look at previous output.' %(e.cmd, git_url, job_model.model_id))

    args = (sys.executable, git_python_script)
    print("$ %s %s" % args)
    job_backend.stop(True)

    subprocess.Popen(args, close_fds=True, env=my_env).wait()

def subprocess_call(args):
    print("$ %s" % (' '.join(args), ))
    return subprocess.call(args, stderr=sys.stderr, stdout=sys.stdout)


def start_keras(job_backend):
    # we need to import keras here, so we know which backend is used (and whether GPU is used)
    from keras import backend as K

    # all our shapes are Tensorflow schema. (height, width, channels)
    if hasattr(K, 'set_image_dim_ordering'):
        K.set_image_dim_ordering('tf')

    job_model = job_backend.get_job_model()

    directory = 'aetros-job/%s/%s' % (job_model.model_id, job_model.index)
    if os.path.exists(directory):
        shutil.rmtree(directory)

    ensure_dir(directory)

    log = io.open('aetros-job/%s/%s/output.log' % (job_model.model_id, job_model.index), 'w', encoding='utf8')
    log.truncate()

    from .KerasCallback import KerasCallback
    trainer = Trainer(job_backend)
    keras_logger = KerasCallback(job_backend, job_backend.general_logger_stdout)

    def ctrlc(sig, frame):
        print("signal %s received\n" % str(sig))
        raise KeyboardInterrupt("CTRL-C!")

    signal.signal(signal.SIGINT, ctrlc)

    try:
        print("Setup job")
        keras_model_utils.job_prepare(job_model)
        job_backend.progress(0, job_backend.job['config']['settings']['epochs'])

        print("Start job")
        keras_model_utils.job_start(job_backend, trainer, keras_logger)

        job_backend.sync_weights()
        job_backend.done()

        print("Done.")
        sys.exit(0)
    except KeyboardInterrupt:
        if job_backend.running:
            job_backend.set_status('ABORTED')
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
