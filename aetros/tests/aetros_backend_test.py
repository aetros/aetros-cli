import unittest

from aetros.backend import parse_message

class TestAetrosBackend(unittest.TestCase):

    def test_parser(self):

        buffer, parsed = parse_message('"MY_MESSAGE"')
        self.assertEqual(buffer, '"MY_MESSAGE"')
        self.assertEqual(parsed, [])

        buffer, parsed = parse_message('"MY_MESSAGE"\n')
        self.assertEqual(buffer, '')
        self.assertEqual(parsed, ["MY_MESSAGE"])

        buffer, parsed = parse_message('"MY_MESSAGE"\t23\n')
        self.assertEqual(buffer, '')
        self.assertEqual(parsed, ["MY_MESSAGE", 23])

        buffer, parsed = parse_message('"MY_MESSAGE"\t23')
        self.assertEqual(buffer, '"MY_MESSAGE"\t23')
        self.assertEqual(parsed, [])

        buffer, parsed = parse_message('"MY_MESSAGE"\t"23\\n"')
        self.assertEqual(buffer, '"MY_MESSAGE"\t"23\\n"')
        self.assertEqual(parsed, [])

        buffer, parsed = parse_message('"MY_MESSAGE"\t"23\\n"\n')
        self.assertEqual(buffer, '')
        self.assertEqual(parsed, ['MY_MESSAGE', "23\n"])
