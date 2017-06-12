# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import absolute_import

import os
import time

from keras.optimizers import Adadelta, Adam, Adamax, Adagrad, RMSprop, SGD

from aetros.backend import JobImage

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
from io import BytesIO

import PIL.Image
import math
from keras.callbacks import Callback
from keras import backend as K
import keras.layers.convolutional

import numpy as np

from aetros.utils.image import get_layer_vis_square, get_layer_vis_square_raw, get_image_tales
from .keras_model_utils import ensure_dir, get_total_params
import six

# compatibility with keras 1.x
def image_data_format():
    if hasattr(K, 'image_data_format'):
        return K.image_data_format()

    if K.image_dim_ordering() == 'th':
        return 'channel_first'
    else:
        return 'channel_last'


class KerasLogger(Callback):
    def __init__(self, trainer, job_backend, general_logger, force_insights=False):
        self.params = {}
        super(KerasLogger, self).__init__()
        self.validation_per_batch = []
        self.insight_layer = []
        self.ins = None
        self.insights_sample_path = None
        self.force_insights = force_insights

        self.trainer = trainer
        self.current = {}
        self.log_epoch = True
        self.confusion_matrix = True

        self.job_backend = job_backend
        self.job_model = job_backend.get_job_model()
        self.general_logger = general_logger
        self._test_with_acc = None
        self.last_batch_time = time.time()
        self.start_time = time.time()
        self.made_batches = 0
        self.batches_per_second = 0
        self.stats = []
        self.last_current = None
        self.filepath_best = self.job_model.get_weights_filepath_best()
        self.filepath_latest = self.job_model.get_weights_filepath_latest()

        ensure_dir(os.path.dirname(self.filepath_best))
        self.data_gathered = False

        self.accuracy_channel = None
        self.all_losses = None
        self.loss_channel = None
        self.learning_rate_channel = None

        self.learning_rate_start = 0

        self.insights_x = None
        self.best_total_accuracy = 0

    def on_train_begin(self, logs={}):
        self.start_time = time.time()
        self.last_batch_time = time.time()
        self.job_backend.set_status('TRAINING')
        self.job_backend.set_info('parameters', get_total_params(self.model))
        self.job_backend.set_info('backend', K.backend())

        if self.model.optimizer and hasattr(self.model.optimizer, 'get_config'):
            config = self.model.optimizer.get_config()
            self.job_backend.set_info('optimizer', type(self.model.optimizer).__name__)
            for i, v in config.items():
                self.job_backend.set_info('optimizer.'+str(i), v)

        self.current['epoch'] = 0
        self.current['started'] = self.start_time
        self.job_backend.set_system_info('current', self.current)

        #compatibility with keras 1.x
        if 'epochs' not in self.params and 'nb_epoch' in self.params:
            self.params['epochs'] = self.params['nb_epoch']
        if 'samples' not in self.params and 'nb_sample' in self.params:
            self.params['samples'] = self.params['nb_sample']

        xaxis = {
            'range': [1, self.params['epochs']],
            'title': u'Epoch ⇢'
        }
        yaxis = {
            'range': [1, 100],
            'title': u'% Accuracy ⇢',
            'dtick': 10
        }

        traces = ['validation', 'training']
        if len(self.model.output_layers) > 1:
            traces = []
            for output in self.model.output_layers:
                traces.append(output.name+'_validation')
                traces.append(output.name+'_training')

        self.accuracy_channel = self.job_backend.create_channel('accuracy', main=True, traces=traces, kpi=True, max_optimization=True, xaxis=xaxis, yaxis=yaxis)
        self.loss_channel = self.job_backend.create_loss_channel('loss', xaxis=xaxis)
        self.learning_rate_channel = self.job_backend.create_channel('learning rate', traces=['start', 'end'], xaxis=xaxis)
        self.job_backend.progress(0, self.params['epochs'])
        if len(self.model.output_layers) > 1:
            loss_traces = []
            for output in self.model.output_layers:
                loss_traces.append(output.name+'_validation')
                loss_traces.append(output.name+'_training')

            self.all_losses = self.job_backend.create_channel('All loss', main=True, xaxis=xaxis, traces=loss_traces)

        images = self.build_insight_images()
        self.job_backend.job_add_insight(0, images, None)

    def on_batch_begin(self, batch, logs={}):

        if not self.data_gathered:
            validation_samples = self.trainer.nb_val_samples
            if 'samples' in self.params:
                training_samples = self.params['samples']
                if hasattr(self.model, 'validation_data') and self.model.validation_data:
                    validation_samples = len(self.model.validation_data[0])
            else:
                training_samples = self.params['steps'] * logs['size']
                if self.trainer.is_generator(self.trainer.data_validation):
                    validation_samples = self.trainer.nb_val_steps * logs['size']
                elif self.trainer.data_validation and isinstance(self.trainer.data_validation, list):
                    validation_samples = len(self.trainer.data_validation[0])

            # we need to do it in on_batch_begin due to the fact that
            # self.model.validation_data is not availabe in on_train_begin
            self.data_gathered = True
            dataset_infos = {}
            dataset_info = {
                'Training': training_samples,
                'Validation': validation_samples,
            }
            dataset_infos['input1'] = dataset_info
            self.job_backend.set_system_info('datasets', dataset_infos)

        if 'nb_batches' not in self.current:
            batch_size = logs['size']
            if 'samples' in self.params:
                nb_batches = math.ceil(self.params['samples'] / batch_size)  # normal nb batches
            else:
                nb_batches = self.params['steps']

            self.current['nb_batches'] = nb_batches
            self.current['batch_size'] = batch_size
            self.job_backend.set_info('Batch size', batch_size)

    def on_batch_end(self, batch, logs={}):
        self.filter_invalid_json_values(logs)
        loss = logs['loss']

        self.validation_per_batch.append(loss)

        self.job_backend.batch(batch, self.current['nb_batches'], logs['size'])

    def write(self, line):
        self.general_logger.write(line)

    def on_epoch_begin(self, epoch, logs={}):
        self.learning_rate_start = self.get_learning_rate()


    def on_epoch_end(self, epoch, logs={}):
        log = logs.copy()

        self.filter_invalid_json_values(log)

        log['created'] = time.time()
        log['epoch'] = epoch + 1
        if 'loss' not in log and len(self.validation_per_batch) > 0:
            log['loss'] = sum(self.validation_per_batch) / float(len(self.validation_per_batch))

        accuracy_log_name = 'acc'
        val_accuracy_log_name = 'val_acc'

        if len(self.model.output_layers) > 1:
            name = self.model.output_layers[0].name
            accuracy_log_name = name+'_acc'
            val_accuracy_log_name = 'val_' + name + '_acc'

        total_accuracy_validation = log.get(val_accuracy_log_name, 0)
        total_accuracy_training = log.get(accuracy_log_name, 0)

        if total_accuracy_validation > self.best_total_accuracy:
            self.best_total_accuracy = total_accuracy_validation
            self.best_epoch = log['epoch']
            try:
                self.model.save_weights(self.filepath_best, overwrite=True)
            except:
                # sometimes hangs with: IOError: Unable to create file (Unable to open file: name = ...
                # without any obvious reason.
                pass

        try:
            self.model.save_weights(self.filepath_latest, overwrite=True)
        except:
            # sometimes hangs with: IOError: Unable to create file (Unable to open file: name = ...
            # without any obvious reason.
            pass

        self.loss_channel.send(log['epoch'], log.get('loss', 0), log.get('val_loss', 0))

        accuracy = [total_accuracy_validation*100, total_accuracy_training*100]
        if len(self.model.output_layers) > 1:
            accuracy = []
            losses = []
            for layer in self.model.output_layers:
                accuracy.append(log.get('val_' + layer.name + '_acc', 0)*100)
                accuracy.append(log.get(layer.name + '_acc', 0)*100)

                losses.append(log.get('val_' + layer.name + '_loss', 0))
                losses.append(log.get(layer.name + '_loss', 0))

            self.all_losses.send(log['epoch'], losses)

        self.accuracy_channel.send(log['epoch'], accuracy)

        self.job_backend.progress(log['epoch'], self.params['epochs'])
        self.send_optimizer_info(log['epoch'])

        if self.log_epoch:
            #todo, multiple outputs
            line = "Epoch %d: loss=%f, acc=%f, val_loss=%f, val_acc=%f\n" % (
                log['epoch'], log['loss'], log.get('acc', 0), log.get('val_loss', 0), total_accuracy_validation, )
            self.general_logger.write(line)

        if self.force_insights or self.job_model.job['insights']:
            # Todo, support multiple inputs
            first_input_layer = self.model.input_layers[0]

            if first_input_layer is not None:

                images = self.build_insight_images()
                # build confusion matrix
                confusion_matrix = self.build_confusion_matrix() if self.confusion_matrix else None

                self.job_backend.job_add_insight(log['epoch'], images, confusion_matrix)

    def send_optimizer_info(self, epoch):
        self.learning_rate_channel.send(epoch, [self.learning_rate_start, self.get_learning_rate()])

    def get_learning_rate(self):

        if hasattr(self.model, 'optimizer'):
            config = self.model.optimizer.get_config()

            if isinstance(self.model.optimizer, Adadelta) or isinstance(self.model.optimizer, Adam) \
                    or isinstance(self.model.optimizer, Adamax) or isinstance(self.model.optimizer, Adagrad)\
                    or isinstance(self.model.optimizer, RMSprop) or isinstance(self.model.optimizer, SGD):
                return config['lr'] * (1. / (1. + config['decay'] * float(K.get_value(self.model.optimizer.iterations))))

            elif 'lr' in config:
                return config['lr']

    def is_image_shape(self, x):
        if len(x.shape) != 3 and len(x.shape) != 2:
            return False

        if len(x.shape) == 2:
            return True

        #  check if it has either 1 or 3 channel
        if K.image_dim_ordering() == 'th':
            return (x.shape[0] == 1 or x.shape[0] == 3)

        if K.image_dim_ordering() == 'tf':
            return (x.shape[2] == 1 or x.shape[2] == 3)

    def get_first_input_sample(self):
        if hasattr(self.model, 'validation_data'):
            input_data_x = []
            for i, input in enumerate(self.model.inputs):
                input_data_x.append([self.model.validation_data[i][0]])
        else:
            if isinstance(self.trainer.data_train['x'], dict):
                input_data_x = []
                for layer in self.model.input_layers:
                    X = self.trainer.data_train['x'][layer.name]
                    if self.trainer.is_generator(X):
                        batch_x, batch_y = next(X)
                        input_data_x.append([batch_x[0]])
                    else:
                        input_data_x.append([X[0]])
            else:
                input_data_x = []
                if self.trainer.is_generator(self.trainer.data_train['x']):
                    batch_x, batch_y = next(self.trainer.data_train['x'])
                    input_data_x.append([batch_x[0]])
                else:
                    for X in self.trainer.data_train['x']:
                        if self.trainer.is_generator(X):
                            batch_x, batch_y = next(X)
                            input_data_x.append([batch_x[0]])
                        else:
                            input_data_x.append([X[0]])

        return input_data_x

    def build_insight_images(self):
        if self.insights_x is None:
            self.insights_x = self.get_first_input_sample()

        images = []

        for i, layer in enumerate(self.model.input_layers):
            x = np.squeeze(self.insights_x[i])
            if self.is_image_shape(x):
                image = self.make_image(x)
                if image:
                    images.append(JobImage(layer.name, image))

        uses_learning_phase = self.model.uses_learning_phase
        inputs = self.model.inputs[:]
        input_data_x_sample = self.insights_x[:]

        if uses_learning_phase:
            inputs += [K.learning_phase()]
            input_data_x_sample += [0.]  # disable learning_phase

        layers = self.model.layers + self.insight_layer

        for layer in layers:
            if isinstance(layer, keras.layers.convolutional.Convolution2D) or isinstance(layer, keras.layers.convolutional.MaxPooling2D):

                fn = K.function(inputs, self.get_layout_output_tensors(layer))
                Y = fn(input_data_x_sample)[0]

                data = np.squeeze(Y)
                if image_data_format() == 'channels_last':
                    data = np.transpose(data, (2, 0, 1))

                image = PIL.Image.fromarray(get_image_tales(data))
                images.append(JobImage(layer.name, image))

                if layer.get_weights():
                    data = layer.get_weights()[0]

                    data = np.transpose(data, (2, 3, 0, 1))

                    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))

                    image = PIL.Image.fromarray(get_image_tales(data))
                    images.append(JobImage(layer.name + '_weights', image, layer.name + ' weights'))

            elif isinstance(layer, keras.layers.ZeroPadding2D) or isinstance(layer, keras.layers.ZeroPadding1D) or isinstance(layer, keras.layers.ZeroPadding3D):
                pass
            elif isinstance(layer, keras.layers.noise.GaussianDropout) or isinstance(layer, keras.layers.noise.GaussianNoise):
                pass
            elif isinstance(layer, keras.layers.Dropout):
                pass
            else:
                outputs = self.get_layout_output_tensors(layer)
                if len(outputs) > 0:
                    fn = K.function(inputs, outputs)
                    Y = fn(input_data_x_sample)[0]
                    Y = np.squeeze(Y)

                    if Y.size == 1:
                        Y = np.array([Y])

                    node = self.job_model.get_model_node(layer.name)

                    image = None
                    if len(Y.shape) > 1:
                        image = PIL.Image.fromarray(get_layer_vis_square(Y))
                    elif len(Y.shape) == 1:
                        if node and node['activationFunction'] == 'softmax':
                            image = self.make_image_from_dense_softmax(Y)
                        else:
                            image = self.make_image_from_dense(Y)

                    if image:
                        images.append(JobImage(layer.name, image))

        return images

    def get_layout_output_tensors(self, layer):
        outputs = []

        if hasattr(layer, 'inbound_nodes'):
            for idx, node in enumerate(layer.inbound_nodes):
                outputs.append(layer.get_output_at(idx))

        return outputs

    def build_confusion_matrix(self):
        confusion_matrix = {}

        model_has_validation_data = hasattr(self.model, 'validation_data') and self.model.validation_data

        if not model_has_validation_data and not self.trainer.data_validation:
            return confusion_matrix

        if len(self.model.output_layers) > 1:
            return confusion_matrix

        first_output_layer = self.model.output_layers[0]

        if 'Softmax' not in str(first_output_layer.output) or len(first_output_layer.output_shape) != 2:
            return confusion_matrix

        input_data_x = None
        input_data_y = []

        if model_has_validation_data:
            input_data_x = []
            for i, layer in enumerate(self.model.input_layers):
                input_data_x.append(self.model.validation_data[i])

            input_data_y = np.squeeze(self.model.validation_data[len(self.model.input_layers)])
        else:
            # model does not have validation_data attribute, which is the case when a
            # generator is given
            if self.trainer.is_generator(self.trainer.data_validation):
                input_data_x = self.trainer.data_validation
            else:
                # it's probably struct of AETROS code generation
                if 'x' in self.trainer.data_validation:
                    if self.trainer.is_generator(self.trainer.data_validation['x']):
                        input_data_x = self.trainer.data_validation['x']
                    elif isinstance(self.trainer.data_validation['x'], dict):
                        for k, X in six.iteritems(self.trainer.data_validation['x']):
                            input_data_x = X

                        if not self.trainer.is_generator(input_data_x):
                            for k, Y in six.iteritems(self.trainer.data_validation['y']):
                                input_data_y = Y

                    elif isinstance(self.trainer.data_validation['x'], list):
                        input_data_x = self.trainer.data_validation['x'][0]
                        if not self.trainer.is_generator(input_data_x):
                            input_data_y = self.trainer.data_validation['y'][0]

        if input_data_x is None:
            return confusion_matrix

        matrix = np.zeros((first_output_layer.output_shape[1], first_output_layer.output_shape[1]))

        if self.trainer.is_generator(input_data_x):
            processed_samples = 0

            while processed_samples < self.trainer.nb_val_samples:
                generator_output = next(input_data_x)
                if len(generator_output) == 2:
                    x, y = generator_output
                    sample_weight = None
                elif len(generator_output) == 3:
                    x, y, sample_weight = generator_output
                else:
                    self.model._stop.set()
                    raise Exception('output of generator should be a tuple '
                                    '(x, y, sample_weight) '
                                    'or (x, y). Found: ' + str(generator_output))

                if type(x) is list:
                    nb_samples = len(x[0])
                elif type(x) is dict:
                    nb_samples = len(list(x.values())[0])
                else:
                    nb_samples = len(x)

                processed_samples += nb_samples

                prediction = self.model.predict_on_batch(x)
                predicted_classes = prediction.argmax(axis=-1)
                expected_classes = y.argmax(axis=-1)

                try:
                    for sample_idx, predicted_class in enumerate(predicted_classes):
                        expected_class = expected_classes[sample_idx]
                        matrix[expected_class, predicted_class] += 1
                except:
                    pass

        else:
            prediction = self.model.predict(
                input_data_x, batch_size=self.job_model.get_batch_size())
            predicted_classes = prediction.argmax(axis=-1)
            expected_classes = np.array(input_data_y).argmax(axis=-1)

            try:
                for sample_idx, predicted_class in enumerate(predicted_classes):
                    expected_class = expected_classes[sample_idx]
                    matrix[expected_class, predicted_class] += 1
            except:
                pass

        confusion_matrix[first_output_layer.name] = matrix.tolist()

        return confusion_matrix

    def filter_invalid_json_values(self, dict):
        for k, v in six.iteritems(dict):
            if isinstance(v, (np.ndarray, np.generic)):
                dict[k] = v.tolist()
            if math.isnan(v) or math.isinf(v):
                dict[k] = -1

    def make_image(self, data):
        from keras.preprocessing.image import array_to_img
        try:
            if len(data.shape) == 2:
                # grayscale image, just add once channel
                data = data.reshape((data.shape[0], data.shape[1], 1))

            image = array_to_img(data)
        except:
            return None

        # image = image.resize((128, 128))

        return image

    def make_image_from_dense_softmax(self, neurons):
        from aetros.utils import array_to_img

        img = array_to_img(neurons.reshape((1, len(neurons), 1)))
        img = img.resize((9, len(neurons) * 8))

        return img

    def make_image_from_dense(self, neurons):
        from aetros.utils import array_to_img
        cols = int(math.ceil(math.sqrt(len(neurons))))

        even_length = cols * cols
        diff = even_length - len(neurons)
        if diff > 0:
            neurons = np.append(neurons, np.zeros(diff, dtype=neurons.dtype))

        img = array_to_img(neurons.reshape((1, cols, cols)))
        img = img.resize((cols * 8, cols * 8))

        return img
