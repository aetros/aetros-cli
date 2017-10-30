import unittest
from collections import OrderedDict
import ruamel.yaml as yaml
from aetros.utils import lose_parameters_to_full

class TestAetrosParametersLose(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super(TestAetrosParameters, self).__init__(methodName)
        self.maxDiff = None

    def assertParametersConverted(self, actual, expected):
        print(yaml.load(expected, Loader=yaml.RoundTripLoader)['parameters'])
        print(lose_parameters_to_full(yaml.load(actual, Loader=yaml.RoundTripLoader)['parameters']))
        self.assertEquals(
            yaml.load(expected, Loader=yaml.RoundTripLoader),
            {'parameters': lose_parameters_to_full(yaml.load(actual, Loader=yaml.RoundTripLoader)['parameters'])}
        )

    def testHyperParameters1(self):
        actual = """
parameters:
    a: 0.2
"""
        expected = """
parameters:
    -  name: a
       type: number
       defaultValue: 0.2
"""

        self.assertParametersConverted(actual, expected)

        actual = """
parameters:
    a: ['a']
    _a: ['a']
"""
        expected = """
parameters:
    -  name: a
       type: choice_string
       defaultValue: 0
       children: [{'value': 'a'}]
    -  name: _a
       type: array
       defaultValue: ['a']
"""

        self.assertParametersConverted(actual, expected)


        actual = """
parameters:
    a: 0.2
    some.name: "peter"
"""
        expected = """
parameters:
    -  name: a
       type: number
       defaultValue: 0.2
    -  name: some.name
       type: string
       defaultValue: "peter"
"""

        self.assertParametersConverted(actual, expected)


    def testHyperParameters7(self):
        actual = """
parameters:
    a: 0.2
    optimizer:
      sgd:
        lr: 0.1
        momentum: 0.5
      adadelta:
        lr:
          start: 0.1
          end: 0.5
"""
        expected = """
parameters:
    -  name: a
       type: number
       defaultValue: 0.2
    -  name: optimizer
       type: choice_group
       defaultValue: 0
       children:
        - name: sgd
          type: group
          children:
            - name: lr
              type: number
              defaultValue: 0.1
            - name: momentum
              type: number
              defaultValue: 0.5
        - name: adadelta
          type: group
          children:
            - name: lr
              type: group
              children:
                - name: start
                  type: number
                  defaultValue: 0.1
                - name: end
                  type: number
                  defaultValue: 0.5
"""

        self.assertParametersConverted(actual, expected);



    def testHyperParameters6(self):
        actual = """
parameters:
    a: 0.2
    optimizer:
      sgd:
        lr: 0.1
        momentum: 0.5
      adadelta:
        lr: 
          start: 0.1
          end: 0.5
        decay: 0.0001
"""
        expected = """
parameters:
    -  name: a
       type: number
       defaultValue: 0.2
    -  name: optimizer
       type: choice_group
       defaultValue: 0
       children:
        - name: sgd
          type: group
          children:
            - name: lr
              type: number
              defaultValue: 0.1
            - name: momentum
              type: number
              defaultValue: 0.5
        - name: adadelta
          type: group
          children: 
            - name: lr
              type: group
              children:
                - name: start
                  type: number
                  defaultValue: 0.1
                - name: end
                  type: number
                  defaultValue: 0.5
            - name: decay
              type: number
              defaultValue: 0.0001 
"""

        self.assertParametersConverted(actual, expected);


    def testHyperParameters5(self):
        actual = """
parameters:
    a: 0.2
    optimizer:
      bla: asa
      sgd:
        lr: 0.1
        momentum: 0.5
      adadelta:
        lr: 1
"""
        expected = """
parameters:
    -  name: a
       type: number
       defaultValue: 0.2
    -  name: optimizer
       type: group
       children:
        - name: bla
          type: string
          defaultValue: asa
        - name: sgd
          type: group
          children:
            - name: lr
              type: number
              defaultValue: 0.1
            - name: momentum
              type: number
              defaultValue: 0.5
        - name: adadelta
          type: group
          children: 
            - name: lr
              type: number
              defaultValue: 1
"""

        self.assertParametersConverted(actual, expected);


    def testHyperParameters4(self):
        actual = """
parameters:
    a: 0.2
    optimizer:
      sgd:
        lr: 0.1
        momentum: 0.5
      adadelta:
        lr: 1
"""
        expected = """
parameters:
    -  name: a
       type: number
       defaultValue: 0.2
    -  name: optimizer
       type: choice_group
       defaultValue: 0
       children:
        - name: sgd
          type: group
          children:
            - name: lr
              type: number
              defaultValue: 0.1
            - name: momentum
              type: number
              defaultValue: 0.5
        - name: adadelta
          type: group
          children: 
            - name: lr
              type: number
              defaultValue: 1
"""

        self.assertParametersConverted(actual, expected)


    def testHyperParameters3(self):
        actual = """
parameters:
    a: 0.2
    dataset: 
        split: 0.5
        names: ["peter", "mowla"]
"""
        expected = """
parameters:
    -  name: a
       type: number
       defaultValue: 0.2
    -  name: dataset
       type: group
       children:
        - name: split
          type: number
          defaultValue: 0.5
        - name: names
          type: choice_string
          defaultValue: 0
          children: 
             - {value: "peter"}
             - {value: "mowla"}
"""

        self.assertParametersConverted(actual, expected);


    def testHyperParameters2(self):
        actual = """
parameters:
    a: 0.2
    list: [4, 2, 6]
    names: ["peter", "mowla"]
"""
        expected = """
parameters:
    -  name: a
       type: number
       defaultValue: 0.2
    -  name: list
       type: choice_number
       defaultValue: 0
       children: 
         - {value: 4}
         - {value: 2}
         - {value: 6}
    -  name: names
       type: choice_string
       defaultValue: 0
       children: 
         - {value: "peter"}
         - {value: "mowla"}
"""

        self.assertParametersConverted(actual, expected);

