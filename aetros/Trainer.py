from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from threading import Lock

def is_generator(obj):
    import inspect

    return obj is not None and (inspect.isgeneratorfunction(obj) or inspect.isgenerator(obj) or hasattr(obj, 'next') or hasattr(obj, '__next__'))


class Trainer():
    """
    :type job_backend : aetros.backend.JobBackend
    :type settings : dict
    """

    def __init__(self, job_backend):
        self.job_backend = job_backend
        self.input_shape = []

        # training sample count per epoch for generator. same name as in keras fit_generator
        self.samples_per_epoch = None #used when simple code uses a generator
        # validation sample count per epoch for generator. same name as in keras fit_generator
        self.nb_val_samples = None #used when simple code uses a generator
        self.nb_val_steps = None

        self.callbacks = [] #used by simple models
        self.classes = None #set by auto_dataset
        self.output_size = None #for code generator output layer, set by auto_dataset
        self.model = None

        self.job_model = job_backend.get_job_model()
        self.settings = self.job_backend.job['config']['settings']

        self.lock = Lock()

    def set_model(self, model):
        self.model = model

    def get_batch_size(self):
        return self.job_backend.get_job_model().get_batch_size()

    def set_generator_validation_nb(self, number):
        """
        sets self.nb_val_samples which is used in model.fit if input is a generator
        :param number:
        :return:
        """

        self.nb_val_samples = number
        diff_to_batch = number % self.get_batch_size()
        if diff_to_batch > 0:
            self.nb_val_samples += self.get_batch_size() - diff_to_batch

        import keras
        if '1' != keras.__version__[0]:
            self.nb_val_samples = self.nb_val_samples // self.get_batch_size()

    def set_generator_training_nb(self, number):
        """
        sets self.samples_per_epoch which is used in model.fit if input is a generator
        :param number:
        :return:
        """

        self.samples_per_epoch = number
        diff_to_batch = number % self.get_batch_size()
        if diff_to_batch > 0:
            self.samples_per_epoch += self.get_batch_size() - diff_to_batch

    def set_status(self, status):
        self.job_backend.set_status(status)

    def set_info(self, name, value):
        self.job_backend.set_info(name, value)

    def has_generator(self, dict):
        for v in dict.values():
            if is_generator(v):
                return True

        return False

    def get_first_generator(self, dict):
        for v in dict.values():
            if is_generator(v):
                return v

        return None

    def set_job_system_info(self, key, value):
        self.job_backend.set_system_info(key, value)

    def set_job_info(self, key, value):
        self.job_backend.set_info(key, value)
