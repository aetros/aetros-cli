import os
import sys

class JobModel:
    def __init__(self, backend, job):
        self.backend = backend
        self.job = job

        network_id = job['networkId']
        job_id = job['id']
        self.id = job['id']

        self.network_id = network_id
        self.job_id = job_id
        pass

    def get_network_h5_path(self):
        return os.getcwd() + '/networks/%s/%s/network.h5' % (self.network_id, self.job_id)

    def get_dataset_dir(self):
        return os.getcwd() + '/networks/%s/%s/datasets' % (self.network_id, self.job_id)

    def get_base_dir(self):
        return os.getcwd() + '/networks/%s/%s' % (self.network_id, self.job_id)

    def get_dataset_downloads_dir(self, dataset):
        return os.getcwd() + '/datasets/%s/datasets_downloads' % (dataset['id'],)

    def get_weights_filepath_latest(self):
        return os.getcwd() + '/weights/%s/latest.hdf5' % (self.job_id,)

    def get_weights_filepath_best(self):
        return os.getcwd() + '/weights/%s/best.hdf5' % (self.job_id,)

    def get_model_provider(self):

        sys.path.append(self.get_base_dir())

        import model_provider
        print "Imported model_provider in %s " % (self.get_base_dir() + '/model_provider.py',)
        sys.path.pop()

        return model_provider

    def sync_weights(self):
        self.backend.job_add_status('status', 'SYNC WEIGHTS')
        print "Sync weights ..."
        self.backend.upload_weights('latest', self.get_weights_filepath_latest())
        self.backend.upload_weights('best', self.get_weights_filepath_best())
        print "Weights synced."

    def network_get_datasets(self, trainer):
        datasets_dir = self.get_dataset_dir()

        datasets = {}

        from aetros.utils import get_option
        from auto_dataset import get_images, read_images_keras_generator, read_images_in_memory

        # load placeholder, auto data
        config = self.job['config']
        for net in config['layer'][0]:
            if 'datasetId' in net and net['datasetId']:

                dataset = config['datasets'][net['datasetId']]
                if not dataset:
                    raise Exception('Dataset of id %s does not exists. Available %s' % (net['datasetId'], ','.join(config['datasets'].keys())))

                if dataset['type'] == 'images_upload' or dataset['type'] == 'images_search':

                    connected_to_net = self.get_connected_network(config['layer'], net)
                    if connected_to_net == None:
                        # this input is not in use, so we dont need to calculate its dataset
                        continue

                    datasets[net['datasetId']] = get_images(config, dataset, net, trainer)

                elif dataset['type'] == 'images_local':

                    all_memory = get_option(dataset['config'], 'allMemory', False, 'bool')

                    if all_memory:
                        datasets[net['datasetId']] = read_images_in_memory(config, dataset, net, trainer)
                    else:
                        datasets[net['datasetId']] = read_images_keras_generator(config, dataset, net, trainer)

                elif dataset['type'] == 'python':
                    name = dataset['id'].replace('/', '__')

                    sys.path.append(datasets_dir)
                    data_provider = __import__(name, '')
                    print "Imported dataset provider in %s " % (datasets_dir + '/' + name + '.py', )
                    sys.path.pop()
                    datasets[dataset['id']] = data_provider.get_data()


        return datasets

    def get_first_input_layer(self):
        config = self.job['config']
        return config['layer'][0][0]

    def get_first_output_layer(self):
        config = self.job['config']
        return config['layer'][-1][0]

    def get_dataset(self, dataset_id):
        return self.job['config']['datasets'][dataset_id]

    def get_connected_network(self, layers, to_net):
        connected_to_net = None
        for nets in layers:
            for net in nets:
                if 'connectedTo' in net:
                    for connectedTo in net['connectedTo']:
                        if connectedTo == to_net['id']:
                            return net

        return connected_to_net