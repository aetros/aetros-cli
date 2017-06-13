import unittest

import sys

import pytest

from aetros.KerasCallback import KerasCallback
from aetros.Trainer import is_generator
from aetros.backend import JobBackend

class TestAetrosCallback(unittest.TestCase):

    def test_set_validation_data(self):
        job_backend = JobBackend('test')
        job_backend.job = {'id': 'test'}

        keras_callback = KerasCallback(job_backend, sys.stdout)

        keras_callback.set_validation_data(([1, 2, 3], [1, 2, 3]))
        self.assertEqual(keras_callback.data_validation_size, 3)

        with pytest.raises(Exception):
            keras_callback.set_validation_data([[1, 2], [1, 2]])

        # aetros format
        keras_callback.set_validation_data({'x': {'input': [1, 2, 3]}, 'y': {'output': [1, 2, 3]}})
        self.assertEqual(keras_callback.data_validation_size, 3)

        keras_callback.set_validation_data({'input': [1, 2, 3], 'output': [1, 2, 3]})
        self.assertEqual(keras_callback.data_validation_size, 3)

        keras_callback.set_validation_data(([1, 2, 3], [1, 2, 3]), 5)
        self.assertEqual(keras_callback.data_validation_size, 3)

        def generator():
            yield ([1, 2], [1, 2])

        self.assertTrue(is_generator(generator))

        with pytest.raises(Exception):
            keras_callback.set_validation_data(generator)

        with pytest.raises(Exception):
            keras_callback.set_validation_data((generator, generator))

        with pytest.raises(Exception):
            keras_callback.set_validation_data([generator, generator], 6)

        keras_callback.set_validation_data(generator, 5)
        self.assertEqual(keras_callback.data_validation_size, 5)

        keras_callback.set_validation_data((generator, generator), 4)
        self.assertEqual(keras_callback.data_validation_size, 4)
