from __future__ import absolute_import
from __future__ import print_function
import argparse


class UploadWeightsCommand:

    def main(self, args):

        from aetros import keras_model_utils

        import aetros.const
        from aetros.backend import JobBackend
        from aetros.logger import GeneralLogger
        from aetros.Trainer import Trainer

        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' upload-weights')
        parser.add_argument('id', nargs='?', help='model name or job id')
        parser.add_argument('--api-key', help="Secure key. Alternatively use API_KEY environment varibale.")
        parser.add_argument('--weights', help="Weights path. Per default we try to find it in the ./weights/ folder.")
        parser.add_argument('--accuracy', help="If you specified model name, you should also specify the accuracy this weights got.")
        parser.add_argument('--latest', action="store_true", help="Instead of best epoch we upload latest weights.")

        parsed_args = parser.parse_args(args)
        job_backend = JobBackend(api_key=parsed_args.api_key)

        if '/' in parsed_args.id and '@' not in parsed_args.id:
            job_backend.create(parsed_args.id)

        job_backend.load(parsed_args.id)

        if job_backend.job is None:
            raise Exception("Job not found")

        job_model = job_backend.get_job_model()

        weights_path = job_model.get_weights_filepath_best()

        if parsed_args.weights:
            weights_path = parsed_args.weights

        print(("Validate weights in %s ..." % (weights_path, )))

        keras_model_utils.job_prepare(job_model)

        general_logger = GeneralLogger()
        trainer = Trainer(job_backend, general_logger)

        job_model.set_input_shape(trainer)

        print("Loading model ...")
        model_provider = job_model.get_model_provider()
        model = model_provider.get_model(trainer)

        loss = model_provider.get_loss(trainer)
        optimizer = model_provider.get_optimizer(trainer)

        print("Compiling ...")
        model_provider.compile(trainer, model, loss, optimizer)

        print(("Validate weights %s ..." % (weights_path,)))
        job_model.load_weights(model, weights_path)
        print("Validated.")

        print("Uploading weights to %s of %s ..." % (job_backend.job_id, job_backend.model_id))

        job_backend.upload_weights('best.hdf5', weights_path, float(parsed_args.accuracy) if parsed_args.accuracy else None)

        print("Done")
