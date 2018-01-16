from __future__ import print_function, division
from __future__ import absolute_import
import simplejson
import os

from aetros.Trainer import is_generator
import six
from six.moves import range

from aetros.utils import invalid_json_values


def ensure_dir(d):
    if not os.path.isdir(d):
        if os.path.isfile(d):  # but a file, so delete it
            print("Deleted", d, "because it was a file, but needs to be an directory.")
            os.remove(d)

        os.makedirs(d)


def get_total_params(model):
    total_params = 0

    flattened_layers = model.flattened_layers if hasattr(
        model, 'flattened_layers') else model.layers

    for i in range(len(flattened_layers)):
        total_params += flattened_layers[i].count_params()

    return total_params


def job_start(job_backend, trainer, keras_callback):
    """
    Starts the training of a job. Needs job_prepare() first.

    :type job_backend: JobBackend
    :type trainer: Trainer
    :return:
    """

    job_backend.set_status('STARTING')
    job_model = job_backend.get_job_model()

    model_provider = job_model.get_model_provider()

    job_backend.set_status('LOAD DATA')
    datasets = job_model.get_datasets(trainer)

    print('trainer.input_shape = %s\n' % (simplejson.dumps(trainer.input_shape, default=invalid_json_values),))
    print('trainer.classes = %s\n' % (simplejson.dumps(trainer.classes, default=invalid_json_values),))

    multiple_inputs = len(datasets) > 1
    insights_x = [] if multiple_inputs else None

    for dataset_name in job_model.get_input_dataset_names():
        dataset = datasets[dataset_name]

        if is_generator(dataset['X_train']):
            batch_x, batch_y = dataset['X_train'].next()

            if multiple_inputs:
                insights_x.append(batch_x[0])
            else:
                insights_x = batch_x[0]
        else:
            if multiple_inputs:
                insights_x.append(dataset['X_train'][0])
            else:
                insights_x = dataset['X_train'][0]

    keras_callback.insights_x = insights_x
    print('Insights sample shape', keras_callback.insights_x.shape)
    keras_callback.write("Possible data keys '%s'\n" % "','".join(list(datasets.keys())))

    data_train = model_provider.get_training_data(trainer, datasets)
    data_validation = model_provider.get_validation_data(trainer, datasets)

    keras_callback.set_validation_data(data_validation, trainer.nb_val_samples)

    trainer.set_status('CONSTRUCT')
    model = model_provider.get_model(trainer)
    trainer.set_model(model)

    trainer.set_status('COMPILING')
    loss = model_provider.get_loss(trainer)
    optimizer = model_provider.get_optimizer(trainer)
    model_provider.compile(trainer, model, loss, optimizer)
    model.summary()

    trainer.callbacks.append(keras_callback)
    model_provider.train(trainer, model, data_train, data_validation)


def job_prepare(job_backend):
    """
    Setups all necessary folder structure so the network can run with datasets code and model_provider.py.
    :type job_model: JobModel
    """

    pass
