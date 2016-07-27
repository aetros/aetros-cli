import json
from threading import Lock

class Trainer():
    def __init__(self, aetros_backend, jobModel, logger):
        self.aetros_backend = aetros_backend
        self.logger = logger
        self.input_shape = []
        self.on_gpu = False

        # for generators
        self.samples_per_epoch = 1
        self.nb_val_samples = 1

        self.data_validation = None
        self.data_train = None

        self.jobModel = jobModel
        self.job = jobModel.job
        self.settings = self.job['config']['settings']
        self.model = None
        self.callbacks = []
        self.lock = Lock()

    def set_model(self, model):
        self.model = model

    def is_generator(self, obj):
        from keras.preprocessing.image import DirectoryIterator
        from aetros.auto_dataset import InMemoryDataGenerator

        if isinstance(obj, DirectoryIterator):
            return True

        if isinstance(obj, InMemoryDataGenerator):
            return True

        return False

    def has_generator(self, dict):
        for v in dict.values():
            if self.is_generator(v):
                return True

        return False

    def get_first_generator(self, dict):
        for v in dict.values():
            if self.is_generator(v):
                return v

        return None

    def set_status(self, status):
        self.lock.acquire()

        print 'Training status changed to %s ' % (status,)
        self.aetros_backend.job_add_status('status', status)

        self.lock.release()

    def set_job_info(self, key, value):
        self.aetros_backend.job_set_info_key(key, value)

