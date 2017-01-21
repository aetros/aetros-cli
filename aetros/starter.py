from __future__ import print_function, division

from __future__ import absolute_import
import io
import logging
import os
import pprint
import signal
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


def start(job_id, dataset_id=None, server_id='local', insights=False, insights_sample_path=None, api_token=None):
    """
    Starts the training process with all logging of a job_id
    """

    job_backend = JobBackend(api_token=api_token)

    if '/' in job_id:
        print("Create job ...")
        job_id = job_backend.create(job_id, server_id=server_id, dataset_id=dataset_id, insights=insights)
        job_backend.load(job_id)
    else:
        job_backend.load(job_id)
        print("Job %s#%d (%s) restarted." % (job_backend.model_id, job_backend.job_index, job_id))

    if not len(job_backend.get_job_model().config):
        raise Exception('Job does not have a configuration. Make sure you created the job via AETROS TRAINER')

    job_model = job_backend.get_job_model()

    #we need to import keras here, so we know which backend is used (and whether GPU is used)
    from keras import backend as K
    job_backend.start()

    ensure_dir('models/%s/%s' % (job_model.model_id, job_model.id))

    log = io.open('models/%s/%s/output.log' % (job_model.model_id, job_model.id), 'w', encoding='utf8')
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
