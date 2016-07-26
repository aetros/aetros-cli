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
from network import ensure_dir

class KerasLogger(Callback):
    def __init__(self, trainer, backend, job_model, general_logger):
        super(KerasLogger, self).__init__()
        self.validation_per_batch = []
        self.ins = None

        self.trainer = None

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

            if first_input != None:

                if self.insight_sample_input_item is None:
                    input_data = self.trainer.data_train['x'][first_input.name]

                    if self.trainer.is_generator(input_data):
                        batch_x, batch_y = input_data.next()
                        self.insight_sample_input_item = batch_x[0]
                    else:
                        self.insight_sample_input_item = input_data[0]

                images = []

                try:
                    image = self.make_image(self.insight_sample_input_item)
                    images.append({
                        'id': input_name,
                        'image': self.to_base64(image)
                    })
                except:
                    pass

                uses_learning_phase = self.model.uses_learning_phase
                inputs = [first_input]
                input_data = [[self.insight_sample_input_item]]

                if uses_learning_phase:
                    inputs += [K.learning_phase()]
                    input_data += [0.]  # disable learning_phase

                for layer in self.model.layers:
                    if isinstance(layer, keras.layers.convolutional.Convolution2D):

                        fn = K.function(inputs, [layer.output])
                        Y = fn(input_data)[0]

                        image = self.make_mosaic(np.squeeze(Y))

                        images.append({
                            'id': layer.name,
                            'image': self.to_base64(image)
                        })

                if len(images) > 0:
                    self.backend.job_add_insight({'epoch': log['epoch']}, images)

    def filterInvalidJsonValues(self, dict):
        for k,v in dict.iteritems():
            if math.isnan(v) or math.isinf(v):
                dict[k] = -1

    def to_base64(self, image):
        buffer = cStringIO.StringIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue())

    def make_image(self, data):
        from keras.preprocessing.image import array_to_img
        try:
            image = array_to_img(data)
        except:
            return None

        # image = image.resize((128, 128))

        return image

    def make_mosaic(self, images):
        from keras.preprocessing.image import array_to_img

        height = images[0].shape[1]
        width = images[0].shape[1]

        cols = int(math.ceil(math.sqrt(len(images))))

        total_height = int(math.ceil(len(images) / cols) * (height + 1))
        total_width = int(cols * (width + 1))

        new_im = PIL.Image.new('L', (total_width, total_height))

        x = 0
        y = 0

        for idx, im in enumerate(images):

            try:
                im = im.reshape((1,width,height))
                im = array_to_img(im)
                new_im.paste(im, (x, y))
            except:
                pass

            x += width + 1
            if idx > 0 and idx % cols == 0:
                y += height + 1
                x = 0

        return new_im