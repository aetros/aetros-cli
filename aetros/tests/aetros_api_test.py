import unittest

from aetros.utils import extract_api_calls

def return_true(call):
    return True

class TestApi(unittest.TestCase):

    def testStdoutReader(self):
        handled_calls, filtered_line, fails = extract_api_calls('blaaa}\n', return_true)
        self.assertEqual(handled_calls, [])
        self.assertEqual(filtered_line, 'blaaa}\n')

        handled_calls, filtered_line, fails = extract_api_calls('{aetros: status, status: foo bar}', return_true)
        self.assertEqual(handled_calls, [])
        self.assertEqual(filtered_line, '{aetros: status, status: foo bar}')

        handled_calls, filtered_line, fails = extract_api_calls('{aetros: status, status: foo bar}\n', return_true)
        self.assertEqual(handled_calls, [{'aetros': 'status', 'status': 'foo bar'}])
        self.assertEqual(filtered_line, '')

        handled_calls, filtered_line, fails = extract_api_calls('2{aetros: status, status: foo bar}\n4', return_true)
        self.assertEqual(handled_calls, [{'aetros': 'status', 'status': 'foo bar'}])
        self.assertEqual(filtered_line, '24')

        handled_calls, filtered_line, fails = extract_api_calls('2{aetros: status, status: foo bar2}\n3{aetros: status, status: foo bar3}\n', return_true)
        self.assertEqual(handled_calls, [{'aetros': 'status', 'status': 'foo bar2'}, {'aetros': 'status', 'status': 'foo bar3'}])
        self.assertEqual(filtered_line, '23')
