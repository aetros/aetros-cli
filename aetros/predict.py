from __future__ import print_function
from __future__ import absolute_import

import json
import numpy as np
import os

from aetros.utils import unpack_full_job_id
from .backend import JobBackend, invalid_json_values

def predict(logger, job_id, file_paths, weights_path=None):

    owner, name, id = unpack_full_job_id(job_id)

    job_backend = JobBackend(model_name=owner+'/'+name)
    job_backend.fetch(id)
    job_backend.load(id)

    job_model = job_backend.get_job_model()
    os.chdir(job_backend.git.work_tree)

    if not weights_path:
        weights_path = job_model.get_weights_filepath_latest()

    from .Trainer import Trainer

    trainer = Trainer(job_backend)
    job_model.set_input_shape(trainer)

    import keras.backend
    if hasattr(keras.backend, 'set_image_dim_ordering'):
        keras.backend.set_image_dim_ordering('tf')

    if hasattr(keras.backend, 'set_image_data_format'):
        keras.backend.set_image_data_format('channels_last')

    job_backend.logger.info("Load model and compile ...")

    model = job_model.get_built_model(trainer)
    trainer.model = model

    from aetros.keras import load_weights
    logger.info('Load weights from ' + weights_path)
    load_weights(model, weights_path)

    inputs = []
    for idx, file_path in enumerate(file_paths):
        inputs.append(job_model.convert_file_to_input_node(file_path, job_model.get_input_node(idx)))

    job_backend.logger.info("Start prediction ...")

    prediction = job_model.predict(trainer, np.array(inputs))

    print(json.dumps(prediction, indent=4, default=invalid_json_values))
