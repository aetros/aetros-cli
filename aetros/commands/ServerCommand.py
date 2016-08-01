import argparse
import os
import sys
import urllib

import cherrypy

from aetros import network

import aetros.const
from aetros.AetrosBackend import AetrosBackend
from aetros.GeneralLogger import GeneralLogger
from aetros.JobModel import JobModel
from aetros.Trainer import Trainer
from aetros.network import ensure_dir

class ServerCommand:

    def main(self, args):

        from aetros.starter import start
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' upload-weights')
        parser.add_argument('id', nargs='?', help='Training id')
        parser.add_argument('--weights', help="Weights path. Per default we try to find it in the ./weights/ folder or download it.")
        parser.add_argument('--latest', action="store_true", help="Instead of best epoch we upload latest weights.")

        parsed_args = parser.parse_args(args)

        if not parsed_args.id:
            parser.print_help()
            sys.exit()

        # model = self.start_model(parsed_args)
        self.start_webserver()

    def start_model(self, parsed_args):

        aetros_backend = AetrosBackend(parsed_args.id)
        job = aetros_backend.get_light_job()
        job_id = job['id']

        aetros_backend.job_id = job_id
        job = aetros_backend.get_job()
        if job is None:
            raise Exception("Training not found")

        job_model = JobModel(aetros_backend, job)

        if parsed_args.weights:
            weights_path = parsed_args.weights
        elif parsed_args.latest:
            weights_path = job_model.get_weights_filepath_latest()
        else:
            weights_path = job_model.get_weights_filepath_best()

        if not os.path.exists(weights_path) or os.path.getsize(weights_path) == 0:
            weight_url = aetros_backend.get_best_weight_url(job_id)
            if not weight_url:
                print("No weights available for this job.")
                exit(1)

            print("Download weights %s to %s .." % (weight_url, weights_path))
            ensure_dir(os.path.dirname(weights_path))

            f = open(weights_path, 'wb')
            f.write(urllib.urlopen(weight_url).read())
            f.close()

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

        print ("Load weights %s ..." %(weights_path,))
        job_model.load_weights(model, weights_path)
        print ("Loaded and ready to go.")

        return model


    def start_webserver(self):
        class WebServer(object):

            @cherrypy.expose
            def index(self):
                return "Hello world!"


        cherrypy.quickstart(WebServer())