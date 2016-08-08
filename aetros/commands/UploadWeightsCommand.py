import argparse
import os
import sys
class UploadWeightsCommand:

    def main(self, args):

        from aetros import network

        import aetros.const
        from aetros.AetrosBackend import AetrosBackend
        from aetros.GeneralLogger import GeneralLogger
        from aetros.JobModel import JobModel
        from aetros.Trainer import Trainer
        from aetros.network import ensure_dir


        from aetros.starter import start
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' upload-weights')
        parser.add_argument('id', nargs='?', help='Network name or training id')
        parser.add_argument('--weights', help="Weights path. Per default we try to find it in the ./weights/ folder.")
        parser.add_argument('--accuracy', help="If you specified network name, you should also specify the accuracy this weights got.")
        parser.add_argument('--latest', action="store_true", help="Instead of best epoch we upload latest weights.")

        parsed_args = parser.parse_args(args)
        aetros_backend = AetrosBackend(parsed_args.id)

        if '/' in parsed_args.id and '@' not in parsed_args.id:
            job_id = aetros_backend.create_job(parsed_args.id)
            aetros_backend.job_id = job_id

        job = aetros_backend.get_light_job()
        job_id = job['id']

        aetros_backend.job_id = job_id
        job = aetros_backend.get_job()
        if job is None:
            raise Exception("Training not found")

        job_model = JobModel(aetros_backend, job)

        weights_path = job_model.get_weights_filepath_best()

        if parsed_args.weights:
            weights_path = parsed_args.weights

        print ("Validate weights in %s ..." % (weights_path, ))

        network.job_prepare(job_model.job)

        general_logger = GeneralLogger(job)
        trainer = Trainer(aetros_backend, job_model, general_logger)

        job_model.set_input_shape(trainer)

        print ("Loading model ...")
        model_provider = job_model.get_model_provider()
        model = model_provider.get_model(trainer)

        loss = model_provider.get_loss(trainer)
        optimizer = model_provider.get_optimizer(trainer)

        print ("Compiling ...")
        model_provider.compile(trainer, model, loss, optimizer)

        print ("Validate weights %s ..." %(weights_path,))
        job_model.load_weights(model, weights_path)
        print ("Validated.")

        print "Uploading weights to %s of %s ..." %(job_id, job['networkId'])

        aetros_backend.upload_weights('best.hdf5', weights_path, float(parsed_args.accuracy))

        print "Done"