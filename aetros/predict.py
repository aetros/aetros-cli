from __future__ import print_function
from __future__ import absolute_import

import io
import json
import tempfile
import urllib

import numpy as np
import os

from aetros import keras_model_utils
from aetros.keras_model_utils import ensure_dir
from .backend import JobBackend, invalid_json_values

def predict(job_id, file_paths, insights=False, weights_path=None, api_key=None):
    print("Prepare model ...")
    job_backend = JobBackend(api_key=api_key)
    job_backend.load(job_id)

    job_model = job_backend.get_job_model()

    log = io.open(tempfile.mktemp(), 'w', encoding='utf8')
    log.truncate()

    keras_model_utils.job_prepare(job_model)

    if not weights_path:
        weight_path = job_model.get_weights_filepath_best()
        if not os.path.exists(weight_path) or os.path.getsize(weight_path) == 0:
            weight_url = job_backend.get_best_weight_url(job_id)
            if not weight_url:
                print("No weights available for this job.")
                exit(1)

            print("Download weights %s to %s .." % (weight_url, weight_path))
            ensure_dir(os.path.dirname(weight_path))

            f = open(weight_path, 'wb')
            f.write(urllib.urlopen(weight_url).read())
            f.close()

    from .logger import GeneralLogger
    from .Trainer import Trainer

    general_logger = GeneralLogger(log, job_backend)
    trainer = Trainer(job_backend, general_logger)
    job_model.set_input_shape(trainer)

    print("Load model and compile ...")

    model = job_model.get_built_model(trainer)

    from aetros.keras import load_weights
    load_weights(model, weights_path)

    inputs = []
    for idx, file_path in enumerate(file_paths):
        inputs.append(job_model.convert_file_to_input_node(file_path, job_model.get_input_node(idx)))

    print("Start prediction ...")

    prediction = job_model.predict(model, np.array(inputs))
    print(json.dumps(prediction, indent=4, default=invalid_json_values))
