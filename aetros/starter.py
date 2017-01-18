from __future__ import print_function, division

from __future__ import absolute_import
import io
import logging
import os
import pprint
import signal
import sys
import traceback

from . import model_utils
from .backend import JobBackend
from .logger import GeneralLogger
from .JobModel import JobModel
from .MonitorThread import MonitoringThread
from .Trainer import Trainer
from .model_utils import ensure_dir
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

        print("Job  %s#%d (%s) created and started. Open http://%s/trainer/app#/training=%s to monitor the training." %
              (job_backend.model_id, job_backend.job_index, job_backend.job_id, job_backend.host, job_id))
    else:
        job_backend.load(job_id)
        print("Job %s#%d (%s) restarted. Open http://%s/trainer/app#/job=%s to monitor the job." %
              (job_backend.model_id, job_backend.job_index, job_id, job_backend.host, job_id))

    if not len(job_backend.get_job_model().config):
        raise Exception('Job does not have a configuration. Make sure you created the job via AETROS TRAINER')

    job_model = job_backend.get_job_model()
    print("start model ...")
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

    monitoringThread = MonitoringThread(job_backend, trainer)
    monitoringThread.daemon = True
    monitoringThread.start()
    model_utils.collect_system_information(trainer)

    def ctrlc(sig, frame):
        print("signal %s received\n" % id)
        raise KeyboardInterrupt("CTRL-C!")

    signal.signal(signal.SIGINT, ctrlc)

    try:
        print("Setup job")
        model_utils.job_prepare(job_model)

        print("Start job")
        model_utils.job_start(job_model, trainer, keras_logger, general_logger)

        job_backend.sync_weights()
        job_backend.stop()
        job_backend.post('job/stopped', json={'id': job_model.id, 'status': 'DONE'})

        print("done.")
        sys.exit(0)
    except KeyboardInterrupt:
        trainer.set_status('STOPPING')
        print('Early stopping ...')

        if job_backend.stop_requested:
            print(' ... stop requested through trainer.')

        if trainer.model:
            trainer.model.stop_training = True

        monitoringThread.stop()
        job_backend.sync_weights()
        job_backend.stop()
        job_backend.post('job/aborted', json={'id': job_model.id})
        print("out.")
        sys.exit(1)
    except Exception as e:
        print("Crashed ...")

        if trainer.model:
            trainer.model.stop_training = True

        log.write(six.text_type(traceback.format_exc()))
        logging.error(traceback.format_exc())

        monitoringThread.stop()
        job_backend.stop()
        job_backend.post('job/crashed', json={'id': job_model.id, 'error': e.message})
        print("out.")
        raise e
