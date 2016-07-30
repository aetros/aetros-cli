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


def predict(job_id, file_path, insights=False):
    print("Prepare network ...")
    aetros_backend = AetrosBackend(job_id)

    job = aetros_backend.get_light_job()
    job_id = job['id']

    if job == 'Job not found':
        raise Exception('Job not found. Have you configured your token correctly?')

    if not isinstance(job, dict):
        raise Exception('Job does not exist. Make sure you created the job via AETROS TRAINER')

    if not len(job['config']):
        raise Exception('Job does not have a configuration. Make sure you created the job via AETROS TRAINER')

    network_id = job['networkId']
    log = io.open(tempfile.mktemp(), 'w', encoding='utf8')
    log.truncate()

    ensure_dir('networks/%s/%s' % (network_id, job_id))
    network.job_prepare(job)

    job_model = JobModel(aetros_backend, job)

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

    first_input_layer = job_model.get_first_input_layer()

    size = (first_input_layer['width'], first_input_layer['height'])
    if first_input_layer['inputType'] == 'image':
        trainer.input_shape = (1, size[0], size[1])
        grayscale = True
    elif first_input_layer['inputType'] == 'image_rgb':
        grayscale = False
        trainer.input_shape = (3, size[0], size[1])
    else:
        trainer.input_shape = (size[0] * size[1],)
        grayscale = True

    first_output_layer = job_model.get_first_output_layer()
    output_dataset = job_model.get_dataset(first_output_layer['datasetId'])

    if output_dataset:
        if 'classes' in job['info']:
            trainer.output_size = len(job['info']['classes'])

    model_provider = job_model.get_model_provider()
    model = model_provider.get_model(trainer)

    loss = model_provider.get_loss(trainer)
    optimizer = model_provider.get_optimizer(trainer)
    model_provider.compile(trainer, model, loss, optimizer)
    model.load_weights(weight_path)

    if 'http://' in file_path or 'https://' in file_path:
        local_image_path = tempfile.mktemp()
        print("Download input ...")
        f = open(local_image_path, 'wb')
        f.write(urllib.urlopen(file_path).read())
        f.close()
    else:
        local_image_path = file_path

    image = Image.open(local_image_path)
    image = image.resize(size, Image.ANTIALIAS)

    if grayscale:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image = np.asarray(image, dtype='float32')

    if len(trainer.input_shape) > 1:
        # RGB: height, width, channel -> channel, height, width
        image = image.transpose(2, 0, 1)
    else:
        # L: height, width => height*width
        image = image.reshape(size[0] * size[1])

    image = image / 255

    input = {}
    for input_layer in model.input_layers:
        input[input_layer.name] = np.array([image])

    print("Start prediction ...")
    prediction = model.predict(input)

    output = dict(zip(job['info']['classes'], prediction[0].tolist()))
    output = sorted(output.items(), reverse=True, key=lambda (k, v): v)
    print(json.dumps(output, indent=4))