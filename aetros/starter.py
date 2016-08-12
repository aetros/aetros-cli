from __future__ import print_function, division

import io
import logging
import os
import pprint
import signal
import sys
import traceback

import network
from AetrosBackend import AetrosBackend
from GeneralLogger import GeneralLogger
from JobModel import JobModel
from MonitorThread import MonitoringThread
from Trainer import Trainer
from network import ensure_dir


def start(job_id, dataset_id=None, server_id='local', insights=False, insights_sample_path=None):
    """
    Starts the training process with all logging of a job_id
    """

    aetros_backend = AetrosBackend(job_id)

    if '/' in job_id:
        print("...")
        job_id = aetros_backend.create_job(job_id, server_id=server_id, dataset_id=dataset_id, insights=insights)
        if job_id is None:
            exit(1)

        print("Training '%s' created and started. Open http://%s/trainer/app?training=%s to monitor the training." %
              (job_id, aetros_backend.host, job_id))
    else:
        print("Training '%s' restarted. Open http://%s/trainer/app?training=%s to monitor the training." %
              (job_id, aetros_backend.host, job_id))

    job = aetros_backend.get_job()

    if job is None or job == 'Job not found':
        raise Exception('Training not found. Have you configured your token correctly?')

    if not isinstance(job, dict):
        raise Exception('Training does not exist. Make sure you created the job via AETROS TRAINER')

    if not len(job['config']):
        raise Exception('Training does not have a configuration. Make sure you created the job via AETROS TRAINER')

    network_id = job['networkId']

    aetros_backend.job_started(job_id, os.getpid())

    ensure_dir('networks/%s/%s' % (network_id, job_id))

    log = io.open('networks/%s/%s/network.log' % (network_id, job_id), 'w', encoding='utf8')
    log.truncate()

    job_model = JobModel(aetros_backend, job)
    general_logger = GeneralLogger(job, log, aetros_backend)

    print("start network ...")

    from KerasLogger import KerasLogger
    trainer = Trainer(aetros_backend, job_model, general_logger)
    keras_logger = KerasLogger(trainer, aetros_backend, job_model, general_logger)
    keras_logger.insights_sample_path = insights_sample_path
    trainer.callbacks.append(keras_logger)

    sys.stdout = general_logger
    sys.stderr = general_logger

    job['running'] = True

    monitoringThread = MonitoringThread(aetros_backend, trainer)
    monitoringThread.daemon = True
    monitoringThread.start()
    network.collect_system_information(trainer)

    def ctrlc(sig, frame):
        print("signal %s received\n" % id)
        raise KeyboardInterrupt("CTRL-C!")

    signal.signal(signal.SIGINT, ctrlc)

    try:
        print("Setup training")
        network.job_prepare(job)

        print("Start training")
        network.job_start(job_model, trainer, keras_logger, general_logger)

        job['running'] = False
        job_model.sync_weights()
        aetros_backend.stop_syncer()
        aetros_backend.post('job/stopped', json={'id': job_model.id, 'status': 'DONE'})

        print("done.")
        sys.exit(0)
    except KeyboardInterrupt:
        trainer.set_status('STOPPING')
        print('Early stopping ...')

        if aetros_backend.stop_requested:
            print(' ... stop requested through trainer.')

        if trainer.model:
            trainer.model.stop_training = True

        monitoringThread.stop()
        job_model.sync_weights()
        aetros_backend.stop_syncer()
        aetros_backend.post('job/stopped', json={'id': job_model.id, 'status': 'EARLY STOP'})
        print("out.")
        sys.exit(1)
    except Exception as e:
        print("Crashed ...")

        if trainer.model:
            trainer.model.stop_training = True

        log.write(unicode(traceback.format_exc()))
        logging.error(traceback.format_exc())

        monitoringThread.stop()
        aetros_backend.stop_syncer()
        aetros_backend.post('job/stopped', json={'id': job_model.id, 'status': 'CRASHED', 'error': e.message})
        print("out.")
        raise e
