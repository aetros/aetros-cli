from __future__ import absolute_import
import argparse
import os
import sys
import aetros.const

class PredictCommand:

    def __init__(self, logger):
        self.logger = logger

    def main(self, args):

        from aetros.predict import predict
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' predict')
        parser.add_argument('id', nargs='?', help='the job id, e.g. peter/mnist/5d0f81d3ea73e8b2da3236c93f502339190c7037')
        parser.add_argument('--weights', help="Weights path. Per default we try to find it in the ./weights/ folder.")
        parser.add_argument('-i', nargs='+', help="Input (path or url). Multiple allowed")
        parser.add_argument('--th', action='store_true', help="Uses Theano instead of Tensorflow")

        parsed_args = parser.parse_args(args)

        if not parsed_args.id:
            parser.print_help()
            sys.exit()

        if not parsed_args.i:
            parser.print_help()
            sys.exit()

        os.environ['KERAS_BACKEND'] = 'theano' if parsed_args.th else 'tensorflow'
        predict(self.logger, parsed_args.id, parsed_args.i, parsed_args.weights)
