from __future__ import print_function, division

from keras.engine import InputLayer, Merge
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Activation, Dense, Embedding, RepeatVector
from keras.models import Sequential

from aetros import network

from aetros.GeneralLogger import GeneralLogger
from aetros.JobModel import JobModel

from aetros.AetrosBackend import AetrosBackend
from aetros.KerasLogger import KerasLogger
from aetros.MonitorThread import MonitoringThread
from aetros.Trainer import Trainer


class KerasIntegration():
    monitoringThread = None
    job_model = None
    trainer = None

    def __init__(self, network_name, model, network_type='custom', insights=False, confusion_matrix=False,
                 insight_sample=None):
        """

        :type network_name: basestring The actual network name available in AETROS Trainer. Example peter/mnist-cnn
        :type insights: bool
        :type confusion_matrix: bool
        :type insight_sample: basestring|None A path to a sample which is being used for the insights. Default is first sample of data_validation.
        """
        self.confusion_matrix = confusion_matrix
        self.model = model

        if isinstance(model, Sequential) and not model.built:
            raise Exception('Sequential model is not built.')

        self.insight_sample = insight_sample
        self.network_name = network_name
        self.insights = insights
        self.network_type = network_type

        self.aetros_backend = AetrosBackend()

        copy = {'fit': self.model.fit, 'fit_generator': self.model.fit_generator}

        def overwritten_fit(x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
                            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, **kwargs):
            self.aetros_backend.set_status('TRAINING')

            callback = self.setup(x, nb_epoch, batch_size)
            callbacks.append(callback)
            copy['fit'](x, y, batch_size, nb_epoch, verbose, callbacks, validation_split, validation_data, True,
                        class_weight, sample_weight, **kwargs)

            self.end()

        def overwritten_fit_generator(generator, samples_per_epoch, nb_epoch,
                                      verbose=1, callbacks=[],
                                      validation_data=None, nb_val_samples=None,
                                      class_weight={}, max_q_size=10, nb_worker=1, pickle_safe=False):
            self.aetros_backend.set_status('TRAINING')

            callback = self.setup(generator, nb_epoch)
            self.trainer.nb_val_samples = nb_val_samples
            self.trainer.data_validation = validation_data
            callbacks.append(callback)

            copy['fit_generator'](generator, samples_per_epoch, nb_epoch,
                                  verbose, callbacks,
                                  validation_data, nb_val_samples,
                                  class_weight, max_q_size, nb_worker, pickle_safe)
            self.end()

        self.model.fit = overwritten_fit
        self.model.fit_generator = overwritten_fit_generator

    def setup(self, x=None, nb_epoch=1, batch_size=16):
        graph = self.model_to_graph(self.model)

        from keras.preprocessing.image import Iterator

        if isinstance(x, Iterator):
            batch_size = x.batch_size

        settings = {
            'epochs': nb_epoch,
            'batchSize': batch_size,
            'optimizer': type(self.model.optimizer).__name__ if hasattr(self.model, 'optimizer') else ''
        }

        self.aetros_backend.ensure_network(self.network_name, self.model.to_json(), settings=settings,
                                           network_type=self.network_type, graph=graph)

        self.aetros_backend.create_job(self.network_name, insights=self.insights)
        job = self.aetros_backend.get_light_job()
        self.job_model = JobModel(self.aetros_backend, job)
        general_logger = GeneralLogger(job, aetros_backend=self.aetros_backend)
        self.trainer = Trainer(self.aetros_backend, self.job_model, general_logger)

        self.monitoringThread = MonitoringThread(self.aetros_backend, self.trainer)
        self.monitoringThread.daemon = True
        self.monitoringThread.start()
        network.collect_system_information(self.trainer)

        self.trainer.model = self.model
        self.trainer.data_train = {'x': x}

        self.callback = KerasLogger(self.trainer, self.aetros_backend, self.job_model, general_logger)
        self.callback.log_epoch = False
        self.callback.model = self.model
        self.callback.confusion_matrix = self.confusion_matrix

        return self.callback

    def publish(self):
        graph = self.model_to_graph(self.model)
        self.aetros_backend.ensure_network(self.network_name, self.model.to_json(), network_type=self.network_type,
                                           graph=graph)

    def start(self, nb_epoch=1, nb_sample=1, title="TRAINING"):
        """
        Starts $title
        :return:
        """

        self.setup(nb_epoch)
        self.callback.params['nb_epoch'] = nb_epoch
        self.callback.params['nb_sample'] = nb_sample
        self.callback.on_train_begin()

        return self.callback

    def batch_begin(self, batch, size):
        logs = {
            'batch': batch,
            'size': size,
        }
        self.callback.on_batch_end(batch, logs)

    def batch_end(self, batch, size, loss=0, acc=0):

        logs = {
            'loss': loss,
            'acc': acc,
            'batch': batch,
            'size': size,
        }
        self.callback.on_batch_end(batch, logs)

    def epoch_end(self, epoch, loss=0, val_loss=0, acc=0, val_acc=0):

        """

        :type epoch: integer starting with 0
        """
        logs = {
            'loss': loss,
            'val_loss': val_loss,
            'acc': acc,
            'val_acc': val_acc,
            'epoch': epoch
        }
        self.callback.on_epoch_end(epoch, logs)

    def end(self):
        self.monitoringThread.stop()
        self.job_model.sync_weights()
        self.aetros_backend.set_status('DONE')

    def model_to_graph(self, model):
        graph = {
            'nodes': [],
            'links': [],
            'groups': []
        }

        map = {
            'idx': {},
            'flatten': [],
            'group_pointer': -1
        }

        def layer_to_dict(layer):
            info = {}

            if isinstance(layer, Dropout):
                info['dropout'] = layer.p

            if isinstance(layer, Dense):
                info['neurons'] = layer.output_dim
                info['activaton'] = layer.activation.__name__

            if isinstance(layer, Convolution2D):
                info['receptiveField'] = [layer.nb_col, layer.nb_row]
                info['features'] = layer.nb_filter

            if isinstance(layer, MaxPooling2D):
                info['poolingArea'] = [layer.pool_size[0], layer.pool_size[1]]

            if isinstance(layer, Embedding):
                info['inputDim'] = layer.input_dim
                info['outputDim'] = layer.output_dim
                info['dropout'] = layer.dropout

            if isinstance(layer, Activation):
                info['activaton'] = layer.activation.__name__

            if isinstance(layer, Merge):
                info['mode'] = layer.mode

            if isinstance(layer, RepeatVector):
                info['n'] = layer.n

            if isinstance(layer, InputLayer):
                info['inputShape'] = layer.input_shape

            info['outputShape'] = layer.output_shape

            return {
                'name': layer.name,
                'class': type(layer).__name__,
                'width': 60,
                'height': 40,
                'info': info
            }

        def add_layer(layer):
            graph['nodes'].append(layer_to_dict(layer))
            map['flatten'].append(layer)
            map['idx'][layer.name] = len(graph['nodes']) - 1
            # if map['group_pointer'] >= 0:
            #     graph['groups'][map['group_pointer']].append(len(graph['nodes'])-1)

        def get_idx(layer):
            return map['idx'][layer.name]

        def extract_layers(layers):
            for layer in layers:
                if layer not in map['flatten']:
                    add_layer(layer)
                    if hasattr(layer, 'layers') and isinstance(layer.layers, list):
                        # graph['groups'].append([])
                        # map['group_pointer'] += 1
                        extract_layers(layer.layers)
                        # map['group_pointer'] -= 1
                    else:
                        for inbound_node in layer.inbound_nodes:
                            extract_layers(inbound_node.inbound_layers)

        extract_layers(model.layers)

        # build edges
        for layer in map['flatten']:

            for inbound_node in layer.inbound_nodes:
                for inbound_layer in inbound_node.inbound_layers:
                    graph['links'].append({
                        'source': get_idx(inbound_layer),
                        'target': get_idx(layer),
                    })

            if hasattr(layer, 'layers') and isinstance(layer.layers, list):
                graph['links'].append({
                    'source': get_idx(layer.layers[-1]),
                    'target': get_idx(layer),
                })

        return graph

    def model_to_layers(self, model):
        layers = []

        # from keras.models import Sequential
        # if isinstance(model, Sequential):
        #     for layer in model.layers:
        #         layers[]


        # 'fc': 'Dense',
        # 'conv': 'Convolutional2D',
        # 'pool': 'MaxPooling2D',
        # 'pool_average': 'AveragePooling2D',
        # 'zero_padding': 'ZeroPadding2D',
        # 'upsampling': 'UpSampling2D',
        # 'flatten': 'Flatten',
        # 'merge': 'Merge',

        layer_type_map = {
            'InputLayer': 'fc',
            'Dense': 'fc',
            'Convolution2D': 'conv',
            'MaxPooling2D': 'pool',
            'AveragePooling2D': 'pool_average',
            'ZeroPadding2D': 'zero_padding',
            'UpSampling2D': 'upsampling',
            'Flatten': 'flatten',
            'Merge': 'merge',
        }

        def get_input_layer(layer):
            if isinstance(layer, Activation) or isinstance(layer, Dropout):
                return get_input_layer(layer.inbound_nodes[0].inbound_layers[0])

            return layer

        for keras_layer in model.layers:
            name = type(keras_layer).__name__

            if name in layer_type_map:
                typeStr = layer_type_map[name]
            else:
                typeStr = name

            layer = {
                'id': keras_layer.name,
                'name': keras_layer.name,
                'type': typeStr,
                'connectedTo': [],
                'receptiveField': {'width': 0, 'height': 0},
                'poolingArea': {'width': 0, 'height': 0},
                'padding': [],
                'features': 0,
            }

            if isinstance(keras_layer, Convolution2D):
                layer['receptiveField']['width'] = keras_layer.nb_col
                layer['receptiveField']['height'] = keras_layer.nb_row
                layer['features'] = keras_layer.nb_filter
            if isinstance(keras_layer, MaxPooling2D):
                layer['poolingArea']['width'] = keras_layer.pool_size[0]
                layer['poolingArea']['height'] = keras_layer.pool_size[1]

            if isinstance(keras_layer, InputLayer):
                if len(keras_layer.input_shape) == 4:

                    # grayscale
                    if keras_layer.input_shape[1] == 1:
                        layer['inputType'] = 'image'
                        layer['width'] = keras_layer.input_shape[2]
                        layer['height'] = keras_layer.input_shape[3]

                    elif keras_layer.input_shape[1] == 3:
                        layer['inputType'] = 'image_rgb'
                        layer['width'] = keras_layer.input_shape[2]
                        layer['height'] = keras_layer.input_shape[3]

                elif len(keras_layer.input_shape) == 2:
                    layer['inputType'] = 'list'
                    layer['width'] = keras_layer.input_shape[1]
                    layer['height'] = 1
                else:
                    layer['inputType'] = 'custom'
                    layer['shape'] = keras_layer.input_shape

            if isinstance(keras_layer, Dense):
                layer['weight'] = keras_layer.output_dim

            if isinstance(keras_layer, Dropout):
                layers[-1][0]['dropout'] = keras_layer.p

                continue

            if isinstance(keras_layer, Activation):
                activation_function = str(keras_layer.activation)
                layers[-1][0]['activationFunction'] = activation_function.split(' ')[1]

                continue

            for inbound_node in keras_layer.inbound_nodes:
                for inbound_layer in inbound_node.inbound_layers:
                    inbound_layer = get_input_layer(inbound_layer)
                    layer['connectedTo'].append(inbound_layer.name)

            layers.append([layer])

        return layers
