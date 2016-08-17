import argparse
import sys
import aetros.const

class PredictCommand:

    def main(self, args):

        from aetros.predict import predict
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' predict')
        parser.add_argument('job', nargs='?', help='the job id, e.g. GoVDO1Njm')
        parser.add_argument('--insights', action='store_true', help="activates insights")
        parser.add_argument('--weights', help="Weights path. Per default we try to find it in the ./weights/ folder.")
        parser.add_argument('-i', nargs='+', help="Input (path or url). Multiple allowed")

        parsed_args = parser.parse_args(args)

        if not parsed_args.job:
            parser.print_help()
            sys.exit()

        predict(parsed_args.job, parsed_args.i, parsed_args.insights, parsed_args.weights)
