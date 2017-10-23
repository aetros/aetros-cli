from __future__ import absolute_import
import argparse
import sys
import aetros.const
import os

class StartCommand:
    def __init__(self, logger):
        self.logger = logger

    def main(self, args):
        from aetros.starter import start
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' start')
        parser.add_argument('name', nargs='?', help='the model name, e.g. aetros/mnist-network to start new job, or job id, e.g. user/modelname/0db75a64acb74c27bd72c22e359de7a4c44a20e5 to restart.')
        parser.add_argument('--insights', action='store_true', help="activates insights. Only for simple models.")
        parser.add_argument('--dataset', help="Dataset id when model has placeholders. Only for simple models with placeholders as input/output.")
        parser.add_argument('--gpu', action='store_true', help="Activates GPU if available. Only for Theano models.")
        parser.add_argument('--device', help="Which device index should be used. Default 0 (which means with --gpu => 'gpu0'). Only for Theano models.")
        parser.add_argument('--tf', action='store_true', help="Force TensorFlow as library. Only for simple models.")
        parser.add_argument('--th', action='store_true', help="Force Theano as library. Only for simple models.")
        parser.add_argument('--mp', help="Activates multithreading if available with given thread count. Only when Theano is active.")
        parser.add_argument('--no-hardware-monitoring', action='store_true', help="Deactivates hardware monitoring")
        parser.add_argument('--param', action='append', help="Sets a hyperparameter, example '--param name=value'. Multiple --param allowed.")

        parsed_args = parser.parse_args(args)

        if not parsed_args.name:
            parser.print_help()
            sys.exit()

        flags = os.environ['THEANO_FLAGS'] if 'THEANO_FLAGS' in os.environ else ''
        if parsed_args.gpu:
            if parsed_args.device:
                flags += ",device=gpu" + parsed_args.device
            else:
                flags += ",device=gpu"

        if parsed_args.mp:
            flags += ",openmp=True"
            os.environ['OMP_NUM_THREADS'] = parsed_args.mp

        os.environ['THEANO_FLAGS'] = flags

        if parsed_args.tf:
            os.environ['KERAS_BACKEND'] = 'tensorflow'

        if parsed_args.th:
            os.environ['KERAS_BACKEND'] = 'theano'

        hyperparameter = {}
        if parsed_args.param:
            for param in parsed_args.param:
                if '=' not in param:
                    raise Exception('--param ' + param+' does not contain a =. Please use "--param name=value"')

                name, value = param.split('=')
                hyperparameter[name] = value

        start(self.logger, parsed_args.name,
            hyperparameter=hyperparameter,
            dataset_id=parsed_args.dataset,
            server='local',
            insights=parsed_args.insights
        )
