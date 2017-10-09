from __future__ import absolute_import
from __future__ import print_function

import json
import os
import sys
import tempfile
import urllib

from PIL import Image

import numpy as np
from aetros.Trainer import Trainer
from aetros.keras_model_utils import ensure_dir
from aetros.backend import JobBackend


class JobModel:
    """
    :type job : dict
    """
    def __init__(self, id, job, storage_dir):
        self.id = id
        self.job = job
        self.storage_dir = storage_dir
        self._layers = None
        self._datasets = None

    @property
    def config(self):
        return self.job['config']

    @property
    def model_settings(self):
        return self.job['modelSettings']

    @property
    def insights_enabled(self):
        return self.config['insights']

    def get_model_node(self, name):
        if not isinstance(self.layers, list):
            return None

        for nodes in self.layers:
            if isinstance(nodes, list):
                for node in nodes:
                    if 'varName' in node and node['varName'] == name:
                        return node
                    if node['id'] == name or ('name' in node and node['name'] == name):
                        return node

        return None

    def get_batch_size(self):
        return self.job['config']['batchSize']

    def get_model_h5_path(self):
        return os.path.normpath(os.getcwd() + '/aetros/model.h5')

    def get_layers_path(self):
        return os.path.normpath(os.getcwd() + '/aetros/layer.json')

    def get_dataset_dir(self):
        return os.path.normpath(os.getcwd() + '/aetros/dataset/')

    def get_dataset_downloads_dir(self, dataset):
        return os.path.normpath(self.storage_dir + '/aetros/dataset/%s/datasets_downloads' % (dataset['id'],))

    def get_weights_filepath_latest(self):
        return os.path.normpath(os.getcwd() + '/aetros/weights/latest.hdf5')

    def get_weights_filepath_best(self):
        return os.path.normpath(os.getcwd() + '/aetros/weights/best.hdf5')

    def get_input_names(self):
        names = []
        for node in self.layer[0]:
            names.append(node['varName'])

        return names

    def get_input_dataset_names(self):
        names = []
        for node in self.layers[0]:
            names.append(node['datasetId'])

        return names

    def set_input_shape(self, trainer):

        trainer.input_shape = {}

        for node in self.layers[0]:
            size = (int(node['width']), (node['height']))
            if node['inputType'] == 'image':
                shape = (size[0], size[1], 1)
            elif node['inputType'] == 'image_rgb':
                shape = (size[0], size[1], 3)
            else:
                shape = (size[0] * size[1],)

            if 'varName' in node:
                trainer.input_shape[node['varName']] = shape
            else:
                # older models
                trainer.input_shape = shape

    def is_python_model(self):
        return 'type' in self.job['config'] and self.job['config']['type'] != 'simple'

    def is_keras_model(self):
        return 'type' in self.job['config'] and self.job['config']['type'] == 'simple'

    def get_built_model(self, trainer):
        # its built with model designer
        model_provider = self.get_model_provider()
        model = model_provider.get_model(trainer)

        loss = model_provider.get_loss(trainer)
        optimizer = model_provider.get_optimizer(trainer)

        model_provider.compile(trainer, model, loss, optimizer)

        return model

    def predict(self, trainer, input):
        prediction = trainer.model.predict(input)

        top5 = np.argsort(-prediction[0])[:5]

        result = []
        for i in top5:
            result.append({
                'class': self.get_dataset_class_label(trainer, self.get_first_output_layer(), i),
                'prediction': float(prediction[0][i])
            })

        return result

    def encode_input_to_input_node(self, input, input_node):

        return input

    def convert_file_to_input_node(self, file_path, input_node=None):
        if input_node is None:
            input_node = self.get_input_node(0)

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
                print(("Could not open %s" % (local_path,)))
                return []

            image = image.resize(size, Image.ANTIALIAS)

            from keras.preprocessing.image import img_to_array

            image = self.convert_image_to_node(image, input_node)

            return image

    def convert_image_to_node(self, image, input_node=None):
        from keras.preprocessing.image import img_to_array

        if input_node is None:
            input_node = self.get_input_node(0)

        if input_node['inputType'] == 'image':
            image = image.convert("L")
            image = img_to_array(image)

        elif input_node['inputType'] == 'image_bgr':
            image = image.convert("RGB")
            image = np.asarray(image, dtype='float32')
            image = image[:, :, ::-1].copy()
            image = img_to_array(image)
        else:
            image = image.convert("RGB")
            image = img_to_array(image)

        if 'imageScale' not in input_node:
            input_node['imageScale'] = 255

        if float(input_node['imageScale']) > 0:
            image = image / float(input_node['imageScale'])

        return image

    def get_model_provider(self):

        sys.path.append('./')

        model = __import__('model', '')
        print("Imported model script from %s " % (os.path.abspath('./model.py'),))
        sys.path.pop()

        return model
    
    @property
    def layers(self):
        if self._layers is None:
            with open(self.get_layers_path(), 'r') as f:
                self._layers = json.loads(f.read())

        return self._layers

    @property
    def datasets(self):
        if self._datasets is None:
            self._datasets = {}

            def read_dataset(id):
                with open(self.get_dataset_dir() + id + '.json', 'r') as f:
                    self._datasets[id] = json.loads(f.read())

            for owner in os.listdir(self.get_dataset_dir()):
                path = os.path.normpath(self.get_dataset_dir() + owner + '/dataset/')
                for name in os.listdir(path):
                    if name.endswith('.json'):
                        read_dataset(owner + '/dataset/' + name[:-len('.json')])

                    if os.path.isdir(path + name):
                        for sub in os.listdir(path + name):
                            if sub.endswith('.json'):
                                read_dataset(owner + '/dataset/' + name + '/' + sub[:-len('.json')])

        return self._datasets

    def get_datasets(self, trainer):
        datasets = {}

        from aetros.utils import get_option
        from .auto_dataset import get_images, read_images_keras_generator, read_images_in_memory

        # load placeholder, auto data
        config = self.job['config']
        for layer in self.layers[0]:
            if 'datasetId' in layer and layer['datasetId']:

                if layer['datasetId'] not in self.datasets:
                    raise Exception('Dataset %s not found in datasets %s' % (layer['datasetId'], json.dumps(self.datasets.keys())))

                dataset = self.datasets[layer['datasetId']]
                if not dataset:
                    raise Exception('Dataset of id %s does not exists. Available %s' % (
                    layer['datasetId'], ','.join(list(self.datasets.keys()))))

                if dataset['type'] == 'images_upload' or dataset['type'] == 'images_search':

                    connected_to_layer = self.get_connected_layer(self.layers, layer)
                    if connected_to_layer is None:
                        # this input is not in use, so we dont need to calculate its dataset
                        continue

                    datasets[layer['datasetId']] = get_images(self, dataset, layer, trainer)

                elif dataset['type'] == 'images_local':

                    all_memory = get_option(dataset['config'], 'allMemory', False, 'bool')

                    if all_memory:
                        datasets[layer['datasetId']] = read_images_in_memory(self, dataset, layer, trainer)
                    else:
                        datasets[layer['datasetId']] = read_images_keras_generator(self, dataset, layer, trainer)

                elif dataset['type'] == 'python':
                    name = dataset['id']

                    try:
                        sys.path.append(os.path.abspath('./aetros/dataset/' + os.path.dirname(name)))
                        data_provider = __import__(os.path.basename(name))
                        print("Imported dataset provider from %s " % (os.path.abspath('./aetros/dataset/' + name + '.py'),))

                        import inspect
                        argSpec = inspect.getargspec(data_provider.get_data)

                        if len(argSpec.args) > 0:
                            datasets[dataset['id']] = data_provider.get_data(trainer)
                        else:
                            datasets[dataset['id']] = data_provider.get_data()

                    except:
                        trainer.logger.error('Could not import dataset code: ' + name + ' in ' + os.path.abspath('./aetros/dataset/'))
                        raise
                    finally:
                        sys.path.pop()

        return datasets

    def get_dataset_class_label(self, trainer, output_layer, prediction):
        dataset_id = output_layer['datasetId']
        dataset = self.datasets[dataset_id]

        if not dataset:
            raise Exception('Dataset of id %s does not exists. Available %s' % (
            dataset_id, ','.join(list(self.datasets.keys()))))

        if trainer.classes:
            return trainer.classes[prediction]

        elif dataset['type'] == 'python':
            name = dataset['id']
            datasets_dir = self.get_dataset_dir()
            try:
                sys.path.append(os.path.abspath('./aetros/dataset/' + os.path.dirname(name)))
                data_provider = __import__(os.path.basename(name))

                if hasattr(data_provider, 'get_class_label'):
                    return data_provider.get_class_label(prediction)
                else:
                    return prediction
            except Exception:
                print(("Method get_class_label failed in %s " % (datasets_dir + '/' + name + '.py',)))
                raise
            finally:
                sys.path.pop()


    def get_first_input_layer(self):
        config = self.job['config']
        return self.layers[0][0]

    def get_input_node(self, idx):
        config = self.job['config']
        return self.layers[0][idx]

    def get_first_output_layer(self):
        config = self.job['config']
        return self.layers[-1][0]

    def get_dataset(self, dataset_id):
        return self.datasets[dataset_id]

    def get_connected_layer(self, layers, to_net):
        connected_to_net = None
        for nets in layers:
            for net in nets:
                if 'connectedTo' in net:
                    for connectedTo in net['connectedTo']:
                        if connectedTo == to_net['id']:
                            return net

        return connected_to_net
