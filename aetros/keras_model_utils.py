from __future__ import print_function, division
from __future__ import absolute_import
import json
import os

from aetros.Trainer import is_generator
from .backend import invalid_json_values
import six
from six.moves import range

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


def job_start(job_backend, trainer, keras_logger):
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

    print('trainer.input_shape = %s\n' % (json.dumps(trainer.input_shape, default=invalid_json_values),))
    print('trainer.classes = %s\n' % (json.dumps(trainer.classes, default=invalid_json_values),))

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

    keras_logger.insights_x = insights_x
    print('Insights sample shape', keras_logger.insights_x.shape)
    keras_logger.write("Possible data keys '%s'\n" % "','".join(list(datasets.keys())))

    data_train = model_provider.get_training_data(trainer, datasets)
    data_validation = model_provider.get_validation_data(trainer, datasets)

    keras_logger.set_validation_data(data_validation, trainer.nb_val_samples)

    trainer.set_status('CONSTRUCT')
    model = model_provider.get_model(trainer)
    trainer.set_model(model)

    trainer.set_status('COMPILING')
    loss = model_provider.get_loss(trainer)
    optimizer = model_provider.get_optimizer(trainer)
    model_provider.compile(trainer, model, loss, optimizer)
    model.summary()

    trainer.callbacks.append(keras_logger)
    model_provider.train(trainer, model, data_train, data_validation)


def job_prepare(job_model):
    """
    Setups all necessary folder structure so the network can run with datasets code and model_provider.py.
    :type job_model: JobModel
    """

    path = 'aetros-job/%s/%s' % (job_model.model_id, job_model.index)
    datasets_path = path + '/datasets'
    config = job_model.config

    ensure_dir(path)
    ensure_dir(datasets_path)

    if not os.path.isfile(path + '/model_provider.py'):
        with open(path + '/model_provider.py', 'w+') as f:
            f.write(config['code'])
            f.close()

    inputOutputNodes = config['layer'][0] + config['layer'][-1]

    for net in inputOutputNodes:
        if net['datasetId']:
            if net['datasetId'] not in config['datasets']:
                raise Exception('Could not find dataset %s. You probably have no access. Available %s' % (
                    net['datasetId'], ','.join(list(config['datasets'].keys()))))

            dataset = config['datasets'][net['datasetId']]
            if dataset['type'] == 'python':

                name = dataset['id'].replace('/', '__')
                dataset_path = datasets_path + '/' + name + '.py'

                if not os.path.isfile(dataset_path):
                    with open(dataset_path, 'w+') as f:
                        f.write(dataset['config']['code'])
                        f.close()
