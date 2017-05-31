from __future__ import print_function, division
from __future__ import absolute_import
import os

import numpy

from aetros.backend import start_job
from aetros.KerasLogger import KerasLogger
from aetros.Trainer import Trainer

def optimizer_factory(settings):
    import keras.optimizers

    optimizer = settings['$value']
    optimizer_settings = settings[optimizer]

    if 'sgd' == optimizer:
        return keras.optimizers.SGD(lr=optimizer_settings['learning_rate'] or 0.01, momentum=optimizer_settings['momentum'] or 0, nesterov=optimizer_settings['nesterov'], decay=optimizer_settings['decay'] or 0.0)

    if 'rmsprop' == optimizer:
        return keras.optimizers.RMSprop(lr=optimizer_settings['learning_rate'] or 0.001, rho=optimizer_settings['rho'] or 0.9, epsilon=optimizer_settings['epsilon'] or 1e-08, decay=optimizer_settings['decay'] or 0.0)

    if 'adagrad' == optimizer:
        return keras.optimizers.Adagrad(lr=optimizer_settings['learning_rate'] or 0.01, epsilon=optimizer_settings['epsilon'] or 1e-08, decay=optimizer_settings['decay'] or 0.0)

    if 'adadelta' == optimizer:
        return keras.optimizers.Adadelta(lr=optimizer_settings['learning_rate'] or 1.0, rho=optimizer_settings['rho'] or 0.95, epsilon=optimizer_settings['epsilon'] or 1e-08, decay=optimizer_settings['decay'] or 0.0)

    if 'adam' == optimizer:
        return keras.optimizers.Adam(lr=optimizer_settings['learning_rate'] or 0.001, beta_1=optimizer_settings['beta_1'] or 0.9, beta_2=optimizer_settings['beta_2'] or 0.999, epsilon=optimizer_settings['epsilon'] or 1e-08, decay=optimizer_settings['decay'] or 0.0)

    if 'adamax' == optimizer:
        return keras.optimizers.Adamax(lr=optimizer_settings['learning_rate'] or 0.002, beta_1=optimizer_settings['beta_1'] or 0.9, beta_2=optimizer_settings['beta_2'] or 0.999, epsilon=optimizer_settings['epsilon'] or 1e-08, decay=optimizer_settings['decay'] or 0.0)

def load_weights(model, weights_path):
    from keras import backend as K

    if not os.path.isfile(weights_path):
        raise Exception("File does not exist.")

    import h5py
    f = h5py.File(weights_path, mode='r')

    # new file format
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    if len(layer_names) != len(model.layers):
        print("Warning: Layer count different")

    # we batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]

        layer = model.get_layer(name=name)
        if layer and len(weight_names):
            weight_values = [g[weight_name] for weight_name in weight_names]
            if not hasattr(layer, 'trainable_weights'):
                print("Layer %s (%s) has no trainable weights, but we tried to load it." % (
                name, type(layer).__name__))
            else:
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

                weight_value_tuples += list(zip(symbolic_weights, weight_values))
    K.batch_set_value(weight_value_tuples)

    f.close()


class KerasIntegration():
    def __init__(self, model_name, model, api_key=None, insights=False, confusion_matrix=False,
                 insight_sample=None, job_backend=None):
        """

        :type model_name: basestring The actual model name available in AETROS Trainer. Example peter/mnist-cnn
        :type insights: bool
        :type confusion_matrix: bool
        :type insight_sample: basestring|None A path to a sample which is being used for the insights. Default is first sample of data_validation.
        """
        self.confusion_matrix = confusion_matrix
        self.model = model

        from keras.models import Sequential

        if isinstance(model, Sequential) and not model.built:
            raise Exception('Sequential model is not built.')

        self.insight_sample = insight_sample
        self.model_name = model_name
        self.insights = insights
        self.model_type = 'custom'
        self.job_backend = job_backend
        self.trainer = None
        self.callback = None
        self.insight_layer = []

        if not self.job_backend:
            self.job_backend = start_job(model_name, api_key=api_key)

        copy = {'fit': self.model.fit, 'fit_generator': self.model.fit_generator}

        def overwritten_fit(
                    x=None,
                    y=None,
                    batch_size=32,
                    epochs=1,
                    verbose=1,
                    callbacks=[],
                    validation_split=0.,
                    validation_data=None,
                    shuffle=True,
                    class_weight=None,
                    sample_weight=None,
                    initial_epoch=0,
                    **kwargs):

            callback = self.setup(x, epochs, batch_size)
            callbacks.append(callback)

            setattr(self.model, 'validation_data', validation_data)

            copy['fit'](x,
                    y,
                    batch_size,
                    epochs,
                    verbose,
                    callbacks,
                    validation_split,
                    validation_data,
                    shuffle,
                    class_weight,
                    sample_weight,
                    initial_epoch,
                    **kwargs)

            self.end()

        # def overwritten_fit_generator(generator, samples_per_epoch, nb_epoch,
        #                               verbose=1, callbacks=[],
        #                               validation_data=None, nb_val_samples=None,
        #                               class_weight={}, max_q_size=10, nb_worker=1, pickle_safe=False):
        def overwritten_fit_generator(
                generator,
                steps_per_epoch,
                epochs=1,
                verbose=1,
                callbacks=[],
                validation_data=None,
                validation_steps=None,
                class_weight=None,
                max_q_size=10,
                workers=1,
                pickle_safe=False,
                initial_epoch=0):

            callback = self.setup(generator, epochs)
            self.trainer.nb_val_steps = validation_steps
            self.trainer.data_validation = validation_data
            callbacks.append(callback)

            copy['fit_generator'](
                generator,
                steps_per_epoch,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_data=validation_data,
                validation_steps=validation_steps,
                class_weight=class_weight,
                max_q_size=max_q_size,
                workers=workers,
                pickle_safe=pickle_safe,
                initial_epoch=initial_epoch)
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

        self.job_backend.ensure_model(self.job_backend.model_id, settings=settings, type=self.model_type)

        self.trainer = Trainer(self.job_backend, self.job_backend.general_logger_stdout)

        self.trainer.model = self.model
        self.trainer.data_train = {'x': x}
        self.job_backend.set_graph(graph)

        self.callback = KerasLogger(self.trainer, self.job_backend, self.job_backend.general_logger_stdout, force_insights=self.insights)
        self.callback.log_epoch = False
        self.callback.model = self.model
        self.callback.insight_layer = self.insight_layer
        self.callback.confusion_matrix = self.confusion_matrix

        return self.callback

    def add_insight_layer(self, layer):
        if self.callback:
            self.callback.insight_layer.append(layer)
        else:
            self.insight_layer.append(layer)

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
        self.job_backend.sync_weights()
        self.job_backend.set_status('DONE')

    def model_to_graph(self, model):
        from keras.engine import InputLayer
        from keras.layers import Convolution2D, MaxPooling2D, Dropout, Activation, Dense, Embedding, RepeatVector, Merge

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
                info['dropout'] = layer.rate

            if isinstance(layer, Dense):
                info['neurons'] = layer.units
                info['activaton'] = layer.activation.__name__

            if isinstance(layer, Convolution2D):
                info['receptiveField'] = layer.kernel_size
                info['features'] = layer.filters

            if isinstance(layer, MaxPooling2D):
                info['poolingArea'] = [layer.pool_size[0], layer.pool_size[1]]

            if isinstance(layer, Embedding):
                info['inputDim'] = layer.input_dim
                info['outputDim'] = layer.output_dim

            if isinstance(layer, Activation):
                info['activaton'] = layer.activation.__name__

            if isinstance(layer, Merge):
                info['mode'] = layer.mode

            if isinstance(layer, RepeatVector):
                info['n'] = layer.n

            if isinstance(layer, InputLayer):
                info['inputShape'] = layer.input_shape

            if hasattr(layer, 'output_shape'):
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
                    elif hasattr(layer, 'inbound_nodes'):
                        for inbound_node in layer.inbound_nodes:
                            extract_layers(inbound_node.inbound_layers)

        extract_layers(model.layers)

        # build edges
        for layer in map['flatten']:

            if hasattr(layer, 'inbound_nodes'):
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
        from keras.engine import InputLayer
        from keras.layers import Convolution2D, MaxPooling2D, Dropout, Activation, Dense

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
                layer['receptiveField']['width'] = keras_layer.kernel_size[0]
                layer['receptiveField']['height'] = keras_layer.kernel_size[1]
                layer['features'] = keras_layer.filters
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
                layer['weight'] = keras_layer.units

            if isinstance(keras_layer, Dropout):
                layers[-1][0]['dropout'] = keras_layer.rate

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
