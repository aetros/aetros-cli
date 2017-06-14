from __future__ import absolute_import
from __future__ import print_function
import argparse
import os


class UploadWeightsCommand:

    def main(self, args):

        from aetros import keras_model_utils

        import aetros.const
        from aetros.backend import JobBackend
        from aetros.Trainer import Trainer

        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' upload-weights')
        parser.add_argument('id', help='model name or job id')
        parser.add_argument('weights', help="Weights path")
        parser.add_argument('--api-key', help="Secure key. Alternatively use API_KEY environment variable.")
        parser.add_argument('--kpi', help="You can overwrite or set the KPI for this job")
        parser.add_argument('--latest', action="store_true", help="Instead of best epoch we upload latest weights.")
        parsed_args = parser.parse_args(args)

        if not parsed_args.id or not parsed_args.weights:
            parser.print_help()
            return

        job_backend = JobBackend(api_key=parsed_args.api_key)

        if '/' in parsed_args.id and '@' not in parsed_args.id:
            job_backend.create(parsed_args.id)

        job_backend.load(parsed_args.id)

        if job_backend.job is None:
            raise Exception("Job not found")

        weights_path = parsed_args.weights

        if not os.path.exists(weights_path):
            raise Exception('Weights file does not exist in ' + weights_path)

        print("Uploading weights to %s of %s ..." % (job_backend.job_id, job_backend.model_id))

        job_backend.upload_weights('weights.hdf5', weights_path, float(parsed_args.kpi) if parsed_args.kpi else None)
