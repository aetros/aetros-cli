import os
import sys
import tempfile
import urllib

from PIL import Image

import numpy as np

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

    def get_model_node(self, name):
        if not isinstance(self.job['config']['layer'], list):
            return None

        for nodes in self.job['config']['layer']:
            if isinstance(nodes, list):
                for node in nodes:
                    if 'varName' in node and node['varName'] == name:
                        return node
                    if node['id'] == name or ('name' in node and node['name'] == name):
                        return node

        return None

    def get_batch_size(self):
        return self.job['config']['settings']['batchSize']

    def get_network_h5_path(self):
        return os.getcwd() + '/aetros-cli-data/networks/%s/%s/network.h5' % (self.network_id, self.job_id)

    def get_dataset_dir(self):
        return os.getcwd() + '/aetros-cli-data/networks/%s/%s/datasets' % (self.network_id, self.job_id)

    def get_base_dir(self):
        return os.getcwd() + '/aetros-cli-data/networks/%s/%s' % (self.network_id, self.job_id)

    def get_dataset_downloads_dir(self, dataset):
        return os.getcwd() + '/aetros-cli-data/datasets/%s/datasets_downloads' % (dataset['id'],)

    def get_weights_filepath_latest(self):
        return os.getcwd() + '/aetros-cli-data/weights/%s/latest.hdf5' % (self.job_id,)

    def get_weights_filepath_best(self):
        return os.getcwd() + '/aetros-cli-data/weights/%s/best.hdf5' % (self.job_id,)

    def set_input_shape(self, trainer):

        trainer.input_shape = {}

        for node in self.job['config']['layer'][0]:
            size = (int(node['width']), (node['height']))
            if node['inputType'] == 'image':
                shape = (1, size[0], size[1])
            elif node['inputType'] == 'image_rgb':
                shape = (3, size[0], size[1])
            else:
                shape = (size[0] * size[1],)

            trainer.input_shape[node['varName']] = shape

    def get_built_model(self, trainer):

        if self.job['config']['fromCode']:
            # its built with custom code using KerasIntegration class
            from keras.models import model_from_json
            model = model_from_json(self.job['config']['model'])
            return model
        else:
            # its built with model designer
            model_provider = self.get_model_provider()
            model = model_provider.get_model(trainer)

            loss = model_provider.get_loss(trainer)
            optimizer = model_provider.get_optimizer(trainer)

            model_provider.compile(trainer, model, loss, optimizer)

        return model

    def load_weights(self, model, weights_path=None):
        from keras import backend as K

        if not weights_path:
            weights_path = self.get_weights_filepath_best()

        if not os.path.isfile(weights_path):
            raise Exception("File does not exist.")

        import h5py
        f = h5py.File(weights_path, mode='r')

        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        if len(layer_names) != len(model.layers):
            print ("Warning: Layer count different")

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for k, name in enumerate(layer_names):
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]

            layer = model.get_layer(name=name)
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in weight_names]
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights

                if len(weight_values) != len(symbolic_weights):
                    raise Exception('Layer #' + str(k) +
                                    ' (named "' + layer.name +
                                    '" in the current model) was found to '
                                    'correspond to layer ' + name +
                                    ' in the save file. '
                                    'However the new layer ' + layer.name +
                                    ' expects ' + str(len(symbolic_weights)) +
                                    ' weights, but the saved weights have ' +
                                    str(len(weight_values)) +
                                    ' elements.')

                weight_value_tuples += zip(symbolic_weights, weight_values)
        K.batch_set_value(weight_value_tuples)

        f.close()

    def predict(self, model, input):
        prediction = model.predict(input)

        top5 = np.argsort(-prediction[0])[:5]

        result = []
        for i in top5:
            result.append({
                'class': self.get_dataset_class_label(self.get_first_output_layer(), i),
                'prediction': float(prediction[0][i])
            })

        return result

    def encode_input_to_input_node(self, input, input_node):

        return input

    def convert_file_to_input_node(self, file_path, input_node):
        size = (int(input_node['width']), int(input_node['height']))

        if 'http://' in file_path or 'https://' in file_path:
            local_path = tempfile.mktemp()
            print("Download input ...")
            f = open(local_path, 'wb')
            f.write(urllib.urlopen(file_path).read())
            f.close()
        else:
            local_path = file_path

        if input_node['inputType'] == 'list':
            raise Exception("List input not yet available")
        else:
            try:
                image = Image.open(local_path)
            except:
                print("Could not open %s" % (local_path,))
                return []

            image = image.resize(size, Image.ANTIALIAS)

            if input_node['inputType'] == 'image':
                image = image.convert("L")
                image = np.asarray(image, dtype='float32')
                image = image.reshape((1, size[0], size[1]))

            elif input_node['inputType'] == 'image_bgr':
                image = image.convert("RGB")
                image = np.asarray(image, dtype='float32')
                image = image[:, :, ::-1].copy()
                image = image.transpose(2, 0, 1)
            else:
                image = image.convert("RGB")
                image = np.asarray(image, dtype='float32')
                image = image.transpose(2, 0, 1)

            if 'imageScale' not in input_node:
                input_node['imageScale'] = 255

            if float(input_node['imageScale']) > 0:
                image = image / float(input_node['imageScale'])

            return image

    def get_model_provider(self):

        sys.path.append(self.get_base_dir())

        import model_provider
        print "Imported model_provider in %s " % (self.get_base_dir() + '/model_provider.py',)
        sys.path.pop()

        return model_provider

    def sync_weights(self):
        self.backend.job_add_status('status', 'SYNC WEIGHTS')
        print "Sync weights ..."
        self.backend.upload_weights('best.hdf5', self.get_weights_filepath_best(), with_status=True)
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

                    datasets[net['datasetId']] = get_images(self, dataset, net, trainer)

                elif dataset['type'] == 'images_local':

                    all_memory = get_option(dataset['config'], 'allMemory', False, 'bool')

                    if all_memory:
                        datasets[net['datasetId']] = read_images_in_memory(self, dataset, net, trainer)
                    else:
                        datasets[net['datasetId']] = read_images_keras_generator(self, dataset, net, trainer)

                elif dataset['type'] == 'python':
                    name = dataset['id'].replace('/', '__')

                    sys.path.append(datasets_dir)
                    data_provider = __import__(name, '')
                    print "Imported dataset provider in %s " % (datasets_dir + '/' + name + '.py', )
                    sys.path.pop()
                    datasets[dataset['id']] = data_provider.get_data()


        return datasets

    def get_dataset_class_label(self, output_layer, prediction):
        config = self.job['config']

        dataset_id = output_layer['datasetId']
        dataset = config['datasets'][dataset_id]
        if not dataset:
            raise Exception('Dataset of id %s does not exists. Available %s' % (dataset_id, ','.join(config['datasets'].keys())))

        if 'classes' in self.job['info']:
            return self.job['info']['classes'][prediction]

        elif dataset['type'] == 'python':
            name = dataset['id'].replace('/', '__')

            datasets_dir = self.get_dataset_dir()
            try:
                sys.path.append(datasets_dir)
                data_provider = __import__(name, '')
                sys.path.pop()
                if hasattr(data_provider, 'get_class_label'):
                    return data_provider.get_class_label(prediction)
                else:
                    return prediction
            except Exception as e:
                print("Method get_class_label failed in %s " % (datasets_dir + '/' + name + '.py', ))
                raise e

    def get_first_input_layer(self):
        config = self.job['config']
        return config['layer'][0][0]

    def get_input_node(self, idx):
        config = self.job['config']
        return config['layer'][0][idx]

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