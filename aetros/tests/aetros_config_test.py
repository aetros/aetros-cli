import unittest
from aetros.utils import is_ignored

class TestAetrosDataset(unittest.TestCase):

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


