from __future__ import print_function

import io
import json
import tempfile
import urllib

import numpy as np
import sys
import os

from PIL import Image

from aetros import network
from aetros.network import ensure_dir
from JobModel import JobModel
from AetrosBackend import AetrosBackend


def predict(job_id, file_path, insights=False, weights_path=None):
    print("Prepare network ...")
    aetros_backend = AetrosBackend(job_id)

    job = aetros_backend.get_light_job()
    job_id = job['id']

    log = io.open(tempfile.mktemp(), 'w', encoding='utf8')
    log.truncate()

    network.job_prepare(job)

    job_model = JobModel(aetros_backend, job)

    if not weights_path:
        weight_path = job_model.get_weights_filepath_best()
        if not os.path.exists(weight_path) or os.path.getsize(weight_path) == 0:
            weight_url = aetros_backend.get_best_weight_url(job_id)
            if not weight_url:
                print("No weights available for this job.")
                exit(1)

            print("Download weights %s to %s .." % (weight_url, weight_path))
            ensure_dir(os.path.dirname(weight_path))

            f = open(weight_path, 'wb')
            f.write(urllib.urlopen(weight_url).read())
            f.close()

    from GeneralLogger import GeneralLogger
    from Trainer import Trainer

    general_logger = GeneralLogger(job, log, aetros_backend)
    trainer = Trainer(aetros_backend, job_model, general_logger)

    job_model.set_input_shape(trainer)

    print("Load model and compile ...")
    model_provider = job_model.get_model_provider()
    model = model_provider.get_model(trainer)

    loss = model_provider.get_loss(trainer)
    optimizer = model_provider.get_optimizer(trainer)

    model_provider.compile(trainer, model, loss, optimizer)
    job_model.load_weights(model, weights_path)

    input = job_model.convert_file_to_input_node(file_path, job_model.get_first_input_layer())

    print("Start prediction ...")

    prediction = job_model.predict(model, np.array([input]))
    print(json.dumps(prediction, indent=4))




