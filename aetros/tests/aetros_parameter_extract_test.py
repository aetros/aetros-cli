import unittest

from aetros.utils import extract_parameters


class TestAetrosParametersExtract(unittest.TestCase):

    def testHyperParameters11(self):
        items = [
            {
                'name': 'num',
                'type': 'number',
                'defaultValue': 0.5
            },
            {
                'name': 'string',
                'type': 'string',
                'defaultValue': 'hi'
            }
        ]

        container = extract_parameters(items)
        self.assertEquals({'num': 0.5, 'string': 'hi'}, container)


    def testHyperParameters12(self):

        itemsOri = [
            {
                'name': 'num',
                'type': 'number',
                'defaultValue': 0.5
            },
            {
                'name': 'string',
                'type': 'string',
                'defaultValue': 'hi'
            }
        ]

        items = itemsOri
        container = extract_parameters(items, {'num': '0.11'})
        self.assertEquals({'num': 0.11, 'string': 'hi'}, container)

        container = extract_parameters(items, {'num': 0.111})
        self.assertEquals({'num': 0.111, 'string': 'hi'}, container)

        container = extract_parameters(items, {'string': 2})
        self.assertEquals({'num': 0.5, 'string': '2'}, container)

        container = extract_parameters(items, {'string': 'asdasd'})
        self.assertEquals({'num': 0.5, 'string': 'asdasd'}, container)


    def testHyperParameters(self):
        items = [
            {
                'name': 'choice',
                'type': 'choice_string',
                'children': [
                    {'value': 'a'},
                    {'value': 'b'},
                    {'value': 'c'},
                ]
            }
        ]

        container = extract_parameters(items)
        self.assertEquals({'choice': 'a'}, container)


    def testHyperParameters2(self):
        items = [
            {
                'name': 'choice',
                'type': 'choice_string',
                'defaultValue': 2,
                'children': [
                    {'value': 'a'},
                    {'value': 'b'},
                    {'value': 'c'},
                ]
            }
        ]

        container = extract_parameters(items)
        self.assertEquals({'choice': 'c'}, container)


    def testHyperParameters3(self):
        items = [
            {
                'name': 'choice',
                'type': 'choice_string',
                'defaultValue': 2,
                'children': [
                    {'value': 'a'},
                    {'value': 'b'},
                    {'value': 'c'},
                ]
            }
        ]

        container = extract_parameters(items, {'choice': 'b'})
        self.assertEquals({'choice': 'b'}, container)


    def testHyperParametersNumber(self):
        items = [
            {
                'name': 'choice',
                'type': 'choice_number',
                'children': [
                    {'value': 2},
                    {'value': 3},
                    {'value': 4},
                ]
            }
        ]

        container = extract_parameters(items, [])
        self.assertEquals({'choice': 2}, container)


    def testHyperParametersNumber2(self):
        items = [
            {
                'name': 'choice',
                'type': 'choice_string',
                'defaultValue': 2,
                'children': [
                    {'value': 2},
                    {'value': 3},
                    {'value': 4},
                ]
            }
        ]

        container = extract_parameters(items, [])
        self.assertEquals({'choice': 4}, container)



    def testHyperParametersNumber3(self):
        items = [
            {
                'name': 'choice',
                'type': 'choice_string',
                'defaultValue': 2,
                'children': [
                    {'value': 2},
                    {'value': 3},
                    {'value': 4},
                ]
            }
        ]

        container = extract_parameters(items, {'choice': 3})
        self.assertEquals({'choice': 3}, container)
