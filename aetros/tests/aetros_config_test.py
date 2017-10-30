import unittest
from aetros.utils import is_ignored, flatten_parameters, read_parameter_by_path


class TestAetrosDataset(unittest.TestCase):

    def testReadParameter(self):

        self.assertEqual(read_parameter_by_path({
            'lr': 0.5
        }, 'lr'), 0.5)

        self.assertEqual(read_parameter_by_path({
            'lr': "peter"
        }, 'lr'), "peter")

        self.assertEqual(read_parameter_by_path({
            'dataset': {
                'split': 0.1,
                'validation': 'jo'
            }
        }, 'dataset.split'), 0.1)

        self.assertEqual(read_parameter_by_path({
            'dataset': {
                'split': 0.1,
                'validation': 'jo'
            }
        }, 'dataset'), {
            'split': 0.1,
            'validation': 'jo'
        })

        choice_group = {
            'optimizer': {
                '$value': 'sgd',
                'sgd': {
                    'lr': 0.1
                }
            }
        }
        self.assertEqual(read_parameter_by_path(choice_group, 'optimizer'), 'sgd')
        self.assertEqual(read_parameter_by_path(choice_group, 'optimizer', return_group=True), {'lr': 0.1})
        self.assertEqual(read_parameter_by_path(choice_group, 'optimizer.lr'), 0.1)

    def testFlattenParameters(self):
        self.assertEqual(flatten_parameters({
            'lr': 0.5
        }), {
            'lr': 0.5
        })

        self.assertEqual(flatten_parameters({
            'list': ['peter', 'mayer']
        }), {
            'list': 'peter'
        })

        self.assertEqual(flatten_parameters({
            '_list': ['peter', 'mayer']
        }), {
            '_list': ['peter', 'mayer']
        })

        self.assertEqual(flatten_parameters({
            'optimizer': {
                'lr': 0.5
            }
        }), {
            'optimizer.lr': 0.5
        })

        self.assertEqual(flatten_parameters({
            'optimizer': {
                '$value': 'sgd',
                'sgd':  {
                    'lr': 0.5
                },
                'adadelta':  {
                    'lr': 0.5
                }
            }
        }), {
            'optimizer': "sgd",
            'optimizer.lr': 0.5,
        })

        self.assertEqual(flatten_parameters({
            'optimizer': {
                'sgd':  {
                    'lr': 0.5
                },
                'adadelta':  {
                    'lr': 0.6
                }
            }
        }), {
            'optimizer.sgd.lr': 0.5,
            'optimizer.adadelta.lr': 0.6,
        })

    def assertIgnored(self, path, patterns):
        self.assertTrue(is_ignored(path, patterns))

    def assertNotIgnored(self, path, patterns):
        self.assertFalse(is_ignored(path, patterns))

    def test_ignore(self):

        self.assertNotIgnored('script.py', '')
        self.assertNotIgnored('script.py', '!script.py')

        self.assertIgnored('script.py', 'script.py')
        self.assertIgnored('script.py', '*.py')
        self.assertNotIgnored('script.py', 'script.py\n!*.py')
        self.assertIgnored('script.py', 'script.py\n!*.py\npt.py')
        self.assertNotIgnored('script.py', 'script.py\n!*.py\n/pt.py')
        self.assertIgnored('/pt.py', 'script.py\n!*.py\n/pt.py')
        self.assertIgnored('script.py', '/*.py')

        self.assertIgnored('folder/script.py', 'script.py')
        self.assertIgnored('script.py', '/script.py')
        self.assertNotIgnored('folder/script.py', '/script.py')
        self.assertNotIgnored('pa-script.py', '/script.py')
        self.assertIgnored('pa-script.py', 'script.py')

        self.assertNotIgnored('script.py', 'folder/script.py')
        self.assertIgnored('folder/script.py', 'folder/script.py')
        self.assertIgnored('sub/folder/script.py', 'folder/script.py')

        self.assertIgnored('folder/script.py', '/folder/script.py')
        self.assertNotIgnored('sub/folder/script.py', '/folder/script.py')

        self.assertNotIgnored('folder/script.py', '/folder/script.py\n!script.py')
        self.assertIgnored('folder/script.py', '\n!*.py\n/folder/script.py')
        self.assertNotIgnored('folder/script.py', '/folder/script.py\n!*.py')
        self.assertIgnored('folder/script.py', '/folder/script.py\n!script.py\nscript.py')

        self.assertIgnored('folder/script.py', '/*/script.py')
        self.assertIgnored('folder/script.py', '/*er/script.py')

        self.assertNotIgnored('folder/script.py', '/*er2/script.py')
        self.assertIgnored('folder2/script.py', '/*er2/script.py')
        self.assertIgnored('folder/script.py', '/*er2/script.py\nfolder')

        self.assertIgnored('very/deep/datasets/dataset.zip', '*.zip')
        self.assertIgnored('very/deep/datasets/dataset.zip', 'datasets/*')
        self.assertNotIgnored('very/deep/datasets/dataset.zip', 'datasets/*\n!dataset.zip')

        self.assertNotIgnored('very/deep/datasets/dataset.zip', '/very/*/*.zip')
        self.assertIgnored('very/deep/datasets/dataset.zip', '/very/**/*.zip')
        self.assertNotIgnored('very/deep/dataset.zip', '/very/**/*.zip')
        self.assertIgnored('very/deep/dataset.zip', '/very/*/*.zip')
        self.assertIgnored('very/deep/dataset.zip', '/very/')


