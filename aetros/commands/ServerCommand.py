import argparse
import json
import os
import sys
import urllib
from threading import Lock
import time

class ServerCommand:

    model = None
    job_model = None

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' upload-weights')
        parser.add_argument('id', nargs='?', help='Training id')
        parser.add_argument('--weights', help="Weights path. Per default we try to find it in the ./weights/ folder or download it.")
        parser.add_argument('--latest', action="store_true", help="Instead of best epoch we upload latest weights.")
        parser.add_argument('--port', help="Changes port. Default 8000")

        parsed_args = parser.parse_args(args)
        self.lock = Lock()

        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

        if not parsed_args.id:
            parser.print_help()
            sys.exit()

        self.model = self.start_model(parsed_args)
        self.start_webserver(8000 if not parsed_args.port else int(parsed_args.port))

    def start_model(self, parsed_args):
        from aetros import network
        from aetros.AetrosBackend import AetrosBackend
        from aetros.GeneralLogger import GeneralLogger
        from aetros.JobModel import JobModel
        from aetros.Trainer import Trainer
        from aetros.network import ensure_dir

        print ("...")
        self.lock.acquire()
        aetros_backend = AetrosBackend(parsed_args.id)
        job = aetros_backend.get_light_job()
        job_id = job['id']

        aetros_backend.job_id = job_id
        job = aetros_backend.get_job()
        if job is None:
            raise Exception("Training not found")

        self.job_model = JobModel(aetros_backend, job)

        if parsed_args.weights:
            weights_path = parsed_args.weights
        elif parsed_args.latest:
            weights_path = self.job_model.get_weights_filepath_latest()
        else:
            weights_path = self.job_model.get_weights_filepath_best()

        print ("Check weights ...")

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

        network.job_prepare(self.job_model.job)

        general_logger = GeneralLogger(job)
        trainer = Trainer(aetros_backend, self.job_model, general_logger)

        self.job_model.set_input_shape(trainer)

        print ("Loading model ...")
        model_provider = self.job_model.get_model_provider()
        model = model_provider.get_model(trainer)

        loss = model_provider.get_loss(trainer)
        optimizer = model_provider.get_optimizer(trainer)

        print ("Compiling ...")
        model_provider.compile(trainer, model, loss, optimizer)

        print ("Load weights %s ..." %(weights_path,))
        self.job_model.load_weights(model, weights_path)
        print ("Locked and loaded.")

        self.lock.release()

        return model

    def start_webserver(self, port):
        import cherrypy
        import numpy

        class WebServer(object):
            def __init__(self, lock, job_model, model):
                self.job_model = job_model
                self.model = model
                self.lock = lock

            @cherrypy.expose
            def predict(self, path):
                if not path:
                    return json.dumps({'error': 'not_input_given'})

                self.lock.acquire()
                result = {'times': {}}
                try:
                    print("Start prediction of %s" % (path,))

                    start = time.time()
                    input = self.job_model.convert_file_to_input_node(path, self.job_model.get_first_input_layer())
                    result['times']['prepare_fetch_input'] = time.time() - start

                    start = time.time()
                    prediction = self.job_model.predict(self.model, numpy.array([input]))
                    result['times']['prediction'] = time.time() - start

                    self.lock.release()
                except Exception as e:
                    self.lock.release()
                    return json.dumps({'error': type(e).__name__, 'message': e.message})

                result['prediction'] = prediction

                return json.dumps(result)

        cherrypy.config.update({
            'server.socket_port': port
        })

        cherrypy.quickstart(WebServer(self.lock, self.job_model, self.model))