from __future__ import absolute_import
import argparse
import os
import sys
import aetros.const

class PredictCommand:

    def main(self, args):

        from aetros.predict import predict
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' predict')
        parser.add_argument('job', nargs='?', help='the job id, e.g. GoVDO1Njm')
        parser.add_argument('--api-key', help="Secure key. Alternatively use API_KEY environment varibale.")
        parser.add_argument('--insights', action='store_true', help="activates insights")
        parser.add_argument('--weights', help="Weights path. Per default we try to find it in the ./weights/ folder.")
        parser.add_argument('-i', nargs='+', help="Input (path or url). Multiple allowed")
        parser.add_argument('--tf', action='store_true', help="Uses TensorFlow instead of Theano")

        parsed_args = parser.parse_args(args)

        if not parsed_args.job:
            parser.print_help()
            sys.exit()

        os.environ['KERAS_BACKEND'] = 'tensorflow' if parsed_args.tf else 'theano'
        predict(parsed_args.job, parsed_args.i, parsed_args.insights, parsed_args.weights, api_key=parsed_args.api_key)
