import unittest

from aetros.Trainer import is_generator

class TestIterator():

    def __iter__(self):
        return self

    def next(self):
        return ([], [])

    def __next__(self):
        return self.next()


class TestAetrosDataset(unittest.TestCase):

    def test_is_generator(self):

        gen = TestIterator()
        self.assertTrue(is_generator(gen))

        def generator():
            yield ([1, 2], [1, 2])

        self.assertTrue(is_generator(generator))