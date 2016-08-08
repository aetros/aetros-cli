from __future__ import division

import base64
import os
import time
import cStringIO

import PIL.Image
import math
from keras.callbacks import Callback
from keras import backend as K
import keras.layers.convolutional

import numpy as np

from aetros.utils.image import get_layer_vis_square
from network import ensure_dir

class KerasLogger(Callback):
    def __init__(self, trainer, backend, job_model, general_logger):
        super(KerasLogger, self).__init__()
        self.validation_per_batch = []
        self.ins = None

        self.trainer = None
        self.insights_sample_path = None

        self.trainer = trainer
        self.backend = backend
        self.job_model = job_model
        self.general_logger = general_logger
        self._test_with_acc = None
        self.last_batch_time = time.time()
        self.start_time = time.time()
        self.made_batches = 0
        self.batches_per_second = 0
        self.stats = []
        self.last_current = None
        self.current = {}
        self.filepath_best = job_model.get_weights_filepath_best()
        self.filepath_latest = job_model.get_weights_filepath_latest()

        ensure_dir(os.path.dirname(self.filepath_best))

        self.insight_sample_input_item = None

        self.best_epoch = 0
        self.best_total_accuracy = 0
        self.worst_total_accuracy = 0

    def set_trainer(self, trainer):
        self.trainer = trainer
        self.current['epoch'] = 0
        self.current['started'] = self.start_time
        self.trainer.set_job_info('current', self.current)

    def on_train_begin(self, logs={}):
        self.start_time = time.time()
        self.last_batch_time = time.time()

        # pprint('on_train_begin')
        # pprint(logs)
        # pprint(self.params)

        nb_sample = self.params['nb_sample'] #training samples total
        nb_epoch = self.params['nb_epoch'] #training samples total

        self.current['nb_sample'] = nb_sample
        self.current['nb_epoch'] = nb_epoch

    def on_batch_begin(self, batch, logs={}):

        # pprint('on_batch_begin')
        # pprint(logs)
        # pprint(self.current)

        batch_size = logs['size']
        nb_batches = math.ceil(self.current['nb_sample'] / batch_size) #normal nb batches

        self.current['nb_batches'] = nb_batches
        self.current['batch_size'] = batch_size

    def on_batch_end(self, batch, logs={}):
        loss = logs['loss']
        if math.isnan(loss) or math.isinf(loss):
            loss = -1

        # pprint('on_batch_end %d ' % (batch,))
        # pprint(logs)
        # pprint(self.current)

        self.validation_per_batch.append(loss)

        current_batch = logs['batch']
        current_batch_size = logs['size'] #how many training items in this batch, differs for the last run

        self.made_batches += 1

        time_diff = time.time() - self.last_batch_time

        if time_diff > 1 or batch == self.current['nb_batches']: #only each second or last batch
            self.batches_per_second = self.made_batches / time_diff
            self.made_batches = 0
            self.last_batch_time = time.time()

            nb_sample = self.params['nb_sample'] #training samples total
            nb_epoch = self.params['nb_epoch'] #training samples total
            batch_size = self.current['batch_size'] #normal batch size
            nb_batches = nb_sample / batch_size #normal nb batches

            self.current['batchesPerSecond'] = self.batches_per_second
            self.current['itemsPerSecond'] = self.batches_per_second * current_batch_size

            epochs_per_second = self.batches_per_second / nb_batches #all batches
            self.current['epochsPerSecond'] = epochs_per_second

            made_batches = (self.current['epoch'] * nb_batches) + current_batch
            total_batches = nb_epoch * nb_batches
            needed_batches = total_batches - made_batches
            seconds_per_batch = 1 / self.batches_per_second

            self.current['eta'] = seconds_per_batch * needed_batches

            self.current['currentBatch'] = current_batch
            self.current['currentBatchSize'] = current_batch_size
            elapsed = time.time() - self.start_time
            self.current['elapsed'] = elapsed

            self.trainer.set_job_info('current', self.current)

    def write(self, line):
        self.general_logger.write(line)

    def on_epoch_end(self, epoch, logs={}):
        log = logs.copy()

        self.filterInvalidJsonValues(log)

        log['created'] = time.time()
        log['epoch'] = epoch+1
        log['loss'] = sum(self.validation_per_batch) / float(len(self.validation_per_batch))

        self.validation_per_batch = []
        log['validation_accuracy'] = {}
        log['validation_loss'] = {}
        log['training_loss'] = {}
        log['training_accuracy'] = {}

        trainer = self.trainer
        elapsed = time.time() - self.start_time

        total_loss = 0
        total_accuracy = 0

        for layer in trainer.model.outputs:
            #todo, this is not very generic
            log['validation_loss'][layer.name] = log['val_loss'] #outs[0]
            log['validation_accuracy'][layer.name] = log['val_acc'] #outs[1]

            log['training_loss'][layer.name] = log['loss'] #outs[0]
            log['training_accuracy'][layer.name] = log['acc'] #outs[1]

            total_loss += log['val_loss']
            total_accuracy += log['val_acc']

        if total_accuracy > self.best_total_accuracy:
            self.best_total_accuracy = total_accuracy
            self.best_epoch = log['epoch']
            self.model.save_weights(self.filepath_best, overwrite=True)

        self.model.save_weights(self.filepath_latest, overwrite=True)

        if total_accuracy < self.worst_total_accuracy:
            self.worst_total_accuracy = total_accuracy

        self.current['totalValidationLoss'] = total_loss
        self.current['totalValidationAccuracy'] = total_accuracy
        self.current['totalValidationAccuracyBest'] = self.best_total_accuracy
        self.current['totalValidationAccuracyWorst'] = self.worst_total_accuracy
        self.current['totalValidationAccuracyBestEpoch'] = self.best_epoch

        self.current['totalTrainingLoss'] = log['loss']
        self.current['elapsed'] = elapsed
        self.current['epoch'] = log['epoch']

        self.filterInvalidJsonValues(self.current)

        trainer.set_job_info('current', self.current)

        line = "Epoch %d: loss=%f, acc=%f, val_loss=%f, val_acc=%f\n" % (log['epoch'], log['loss'], log.get('acc'), log['val_loss'], log.get('val_acc'), )
        self.general_logger.write(line)

        self.backend.job_add_status('epoch', log)

        input_name = None

        if self.job_model.job['insights']:

            #Todo, support multiple inputs
            first_input = self.model.inputs[0]
            first_input_layer = self.model.input_layers[0]
            first_output_layer = self.model.output_layers[0]

            if first_input_layer != None:
                input_data_x = self.trainer.data_validation['x'][first_input_layer.name]
                input_data_y = self.trainer.data_validation['y'][first_output_layer.name]
                confusion_matrix = {}

                input_node = self.job_model.get_model_node(first_input_layer.name)
                if self.insight_sample_input_item is None:
                    if self.insights_sample_path:
                        self.insight_sample_input_item = self.job_model.convert_file_to_input_node(self.insights_sample_path, input_node)
                    else:
                        if self.trainer.is_generator(input_data_x):
                            batch_x, batch_y = input_data_x.next()
                            self.insight_sample_input_item = batch_x[0]
                        else:
                            self.insight_sample_input_item = input_data_x[0]

                # build confusion matrix
                node = self.job_model.get_model_node(first_output_layer.name)
                if node and node['classificationMode'] == 'categorical':
                    matrix = np.zeros((first_output_layer.output_shape[1], first_output_layer.output_shape[1]))
                    if self.trainer.is_generator(input_data_x):
                        processed_samples = 0

                        while processed_samples < trainer.nb_val_samples:
                            generator_output = input_data_x.next()
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

                            prediction = trainer.model.predict_on_batch(x)
                            predicted_classes = prediction.argmax(axis=-1)
                            expected_classes = y.argmax(axis=-1)

                            try:
                                for sample_idx, predicted_class in enumerate(predicted_classes):
                                    expected_class = expected_classes[sample_idx]
                                    matrix[expected_class, predicted_class] += 1
                            except:
                                pass

                    else:
                        prediction = trainer.model.predict(input_data_x, batch_size=self.job_model.get_batch_size())
                        predicted_classes = prediction.argmax(axis=-1)
                        expected_classes = input_data_y.argmax(axis=-1)

                        try:
                            for sample_idx, predicted_class in enumerate(predicted_classes):
                                expected_class = expected_classes[sample_idx]
                                matrix[expected_class, predicted_class] += 1
                        except:
                            pass

                    confusion_matrix[first_output_layer.name] = matrix.tolist()

                images = []

                try:
                    image = self.make_image(self.insight_sample_input_item)
                    images.append({
                        'id': input_name,
                        'title': input_name,
                        'image': self.to_base64(image)
                    })
                except:
                    pass

                uses_learning_phase = self.model.uses_learning_phase
                inputs = [first_input]
                input_data_x_sample = [[self.insight_sample_input_item]]

                if uses_learning_phase:
                    inputs += [K.learning_phase()]
                    input_data_x_sample += [0.]  # disable learning_phase

                for layer in self.model.layers:
                    if isinstance(layer, keras.layers.convolutional.Convolution2D) or isinstance(layer, keras.layers.convolutional.MaxPooling2D):

                        fn = K.function(inputs, [layer.output])
                        Y = fn(input_data_x_sample)[0]

                        data = np.squeeze(Y)
                        # print("Layer Activations " + layer.name)
                        image = PIL.Image.fromarray(get_layer_vis_square(data))

                        images.append({
                            'id': layer.name,
                            'type': 'convolution',
                            'title': layer.name,
                            'image': self.to_base64(image)
                        })

                        if hasattr(layer, 'W') and layer.W:
                            # print("Layer Weights " + layer.name)
                            data = layer.W.get_value()
                            image = PIL.Image.fromarray(get_layer_vis_square(data))
                            images.append({
                                'id': layer.name+'_weights',
                                'type': 'convolution',
                                'title': layer.name + ' weights',
                                'image': self.to_base64(image)
                            })


                    if isinstance(layer, keras.layers.Dense):

                        fn = K.function(inputs, [layer.output])
                        Y = fn(input_data_x_sample)[0]

                        node = self.job_model.get_model_node(layer.name)
                        if node and node['activationFunction'] == 'softmax':
                            image = self.make_image_from_dense_softmax(np.squeeze(Y))
                        else:
                            image = self.make_image_from_dense(np.squeeze(Y))

                        images.append({
                            'id': layer.name,
                            'type': 'dense',
                            'title': layer.name,
                            'image': self.to_base64(image)
                        })

                self.backend.job_add_insight({'epoch': log['epoch'], 'confusionMatrix': confusion_matrix}, images)



    def filterInvalidJsonValues(self, dict):
        for k,v in dict.iteritems():
            if math.isnan(v) or math.isinf(v):
                dict[k] = -1

    def to_base64(self, image):
        buffer = cStringIO.StringIO()
        image.save(buffer, format="JPEG", optimize=True, quality=80)
        return base64.b64encode(buffer.getvalue())

    def make_image(self, data):
        from keras.preprocessing.image import array_to_img
        try:
            image = array_to_img(data)
        except:
            return None

        # image = image.resize((128, 128))

        return image

    def make_image_from_dense_softmax(self, neurons):
        from aetros.utils import array_to_img

        img = array_to_img(neurons.reshape((1, len(neurons), 1)))
        img = img.resize((9, len(neurons)*8))

        return img

    def make_image_from_dense(self, neurons):
        from aetros.utils import array_to_img
        cols = int(math.ceil(math.sqrt(len(neurons))))

        even_length = cols*cols
        diff = even_length - len(neurons)
        if diff > 0:
            neurons = np.append(neurons, np.zeros(diff, dtype=neurons.dtype))

        img = array_to_img(neurons.reshape((1, cols, cols)))
        img = img.resize((cols*8, cols*8))

        return img
