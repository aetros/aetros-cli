from __future__ import print_function, division
import json
import os
from pprint import pprint

def ensure_dir(d):
    if not os.path.isdir(d):
        if os.path.isfile(d): #but a file, so delete it
            os.remove(d)

        os.makedirs(d)

def get_total_params(model):
    total_params = 0

    flattened_layers = model.flattened_layers if hasattr(model, 'flattened_layers') else model.layers

    for i in range(len(flattened_layers)):
        total_params += flattened_layers[i].count_params()

    return total_params

def collect_system_information(trainer):
    import psutil

    mem = psutil.virtual_memory()
    trainer.set_job_info('memory_total', mem.total)

    # at this point, theano is already initialised through KerasLogger
    from theano.sandbox import cuda
    trainer.set_job_info('cuda_available', cuda.cuda_available)
    if cuda.cuda_available:
        trainer.on_gpu = cuda.use.device_number is not None
        trainer.set_job_info('cuda_device_number', cuda.active_device_number())
        trainer.set_job_info('cuda_device_name', cuda.active_device_name())
        if cuda.cuda_ndarray.cuda_ndarray.mem_info:
            gpu = cuda.cuda_ndarray.cuda_ndarray.mem_info()
            trainer.set_job_info('cuda_device_max_memory', gpu[1])
            free = gpu[0]/1024/1024/1024
            total = gpu[1]/1024/1024/1024
            used = total-free
            if trainer.on_gpu:
                print("%.2fGB GPU memory used of %.2fGB" %(used, total))

    trainer.set_job_info('on_gpu', trainer.on_gpu)

    import cpuinfo
    cpu = cpuinfo.get_cpu_info()
    trainer.set_job_info('cpu_name', cpu['brand'])
    trainer.set_job_info('cpu', [cpu['hz_actual_raw'][0], cpu['count']])


def job_start(job_model, trainer, keras_logger, general_logger):
    """
    Starts the training of a job. Needs job_prepare() first.

    :param job_model:
    :param trainer:
    :param keras_logger:
    :param general_logger:
    :return:
    """
    trainer.set_status('STARTING')

    model_provider = job_model.get_model_provider()

    trainer.set_status('LOAD DATA')
    datasets = job_model.network_get_datasets(trainer)
    general_logger.write('trainer.input_shape = %s\n' % (json.dumps(trainer.input_shape),))
    general_logger.write('trainer.classes = %s\n' % (json.dumps(trainer.classes),))

    dataset_infos = {}
    for idx, dataset in datasets.iteritems():

        if trainer.is_generator(dataset['X_train']):
            training = trainer.samples_per_epoch
        else:
            training = len(dataset['X_train'])

        if trainer.is_generator(dataset['X_test']):
            validation = trainer.nb_val_samples
        else:
            validation = len(dataset['X_test'])

        dataset_info = {
            'Training': training,
            'Validation': validation,
        }

        dataset_infos[idx] = dataset_info

    trainer.set_job_info('datasets', dataset_infos)
    keras_logger.write("Possible data keys '%s'\n" % "','".join(datasets.keys()))

    data_train = model_provider.get_training_data(trainer, datasets)
    data_validation = model_provider.get_validation_data(trainer, datasets)

    trainer.data_train = data_train
    trainer.data_validation = data_validation

    trainer.set_status('CONSTRUCT')
    model = model_provider.get_model(trainer)
    trainer.set_model(model)

    trainer.set_status('COMPILING')
    loss = model_provider.get_loss(trainer)
    optimizer = model_provider.get_optimizer(trainer)
    model_provider.compile(trainer, model, loss, optimizer)
    model.summary()

    trainer.set_job_info('total_params', get_total_params(model))

    model_provider.train(trainer, model, data_train, data_validation)

def job_prepare(job):
    """
    Setups all necessary folder structure so the network can run with datasets code and model_provider.py.

    :param job: job dict
    :return:
    """

    path = 'aetros-cli-data/networks/%s/%s' % (job['networkId'], job['id'])
    datasets_path = path + '/datasets'
    config = job['config']

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
                raise Exception('Could not find dataset %s. You probably have no access. Available %s' % (net['datasetId'], ','.join(config['datasets'].keys())))

            dataset = config['datasets'][net['datasetId']]
            if dataset['type'] == 'python':

                name = dataset['id'].replace('/', '__')
                dataset_path = datasets_path + '/' + name + '.py'

                if not os.path.isfile(dataset_path):
                    with open(dataset_path, 'w+') as f:
                        f.write(dataset['config']['code'])
                        f.close()
