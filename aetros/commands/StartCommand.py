import argparse
import os
import sys

import aetros.const

class StartCommand:

    def main(self, args):

        from aetros.starter import start
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' start')
        parser.add_argument('network_name', nargs='?', help='the network name, e.g. aetros/mnist-network')
        parser.add_argument('--insights', action='store_true', help="activates insights")
        parser.add_argument('--insights-sample', help="Path or url to the sample being used to generate the insights. Default is first training sample.")
        parser.add_argument('--dataset', help="Dataset id when network has placeholders")
        parser.add_argument('--gpu', action='store_true', help="Activates GPU if available")
        parser.add_argument('--device', help="Which device index should be used. Default 0 (which means with --gpu => 'gpu0')")
        parser.add_argument('--tf', action='store_true', help="Uses TensorFlow instead of Theano")
        parser.add_argument('--mp', help="Activates multithreading if available with given thread count.")

        parsed_args = parser.parse_args(args)

        if not parsed_args.network_name:
            parser.print_help()
            sys.exit()

        import os
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

        start(parsed_args.network_name, dataset_id=parsed_args.dataset, insights=parsed_args.insights, insights_sample_path=parsed_args.insights_sample)
