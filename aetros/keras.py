from __future__ import print_function, division
from __future__ import absolute_import
import os

from aetros.backend import start_job
from aetros.KerasCallback import KerasCallback
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

def model_to_graph(model):
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