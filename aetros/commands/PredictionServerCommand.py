from __future__ import absolute_import
from __future__ import print_function
import argparse
import json
import os
import sys
import urllib
from threading import Lock
import time


class PredictionServerCommand:

    model = None
    job_model = None

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' server')
        parser.add_argument('id', nargs='?', help='job id')
        parser.add_argument('--weights', help="Weights path. Per default we try to find it in the ./weights/ folder or download it.")
        parser.add_argument('--latest', action="store_true", help="Instead of best epoch we upload latest weights.")
        parser.add_argument('--tf', action='store_true', help="Uses TensorFlow instead of Theano")
        parser.add_argument('--port', help="Changes port. Default 8000")
        parser.add_argument('--host', help="Changes host. Default 127.0.0.1")

        parsed_args = parser.parse_args(args)
        self.lock = Lock()

        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

        if not parsed_args.id:
            parser.print_help()
            sys.exit()

        os.environ['KERAS_BACKEND'] = 'tensorflow' if parsed_args.tf else 'theano'

        self.model = self.start_model(parsed_args)
        self.start_webserver('127.0.0.1' if not parsed_args.host else parsed_args.host, 8000 if not parsed_args.port else int(parsed_args.port))

    def start_model(self, parsed_args):
        from aetros import keras_model_utils
        from aetros.backend import JobBackend
        from aetros.logger import GeneralLogger
        from aetros.Trainer import Trainer
        from aetros.keras_model_utils import ensure_dir

        if not parsed_args.id:
            print("No job id given.")
            sys.exit(1)

        print("...")
        self.lock.acquire()
        job_backend = JobBackend(parsed_args.id)
        job_backend.load_light_job()
        self.job_model = job_backend.get_job_model()

        if parsed_args.weights:
            weights_path = parsed_args.weights
        elif parsed_args.latest:
            weights_path = self.job_model.get_weights_filepath_latest()
        else:
            weights_path = self.job_model.get_weights_filepath_best()

        print("Check weights ...")

        if not os.path.exists(weights_path) or os.path.getsize(weights_path) == 0:
            weight_url = job_backend.get_best_weight_url(parsed_args.id)
            if not weight_url:
                print("No weights available for this job.")
                exit(1)

            print(("Download weights %s to %s .." % (weight_url, weights_path)))
            ensure_dir(os.path.dirname(weights_path))

            f = open(weights_path, 'wb')
            f.write(urllib.urlopen(weight_url).read())
            f.close()

        keras_model_utils.job_prepare(self.job_model)

        general_logger = GeneralLogger()
        trainer = Trainer(job_backend, general_logger)

        self.job_model.set_input_shape(trainer)

        print("Loading model ...")
        model = self.job_model.get_built_model(trainer)

        print(("Load weights %s ..." % (weights_path,)))
        self.job_model.load_weights(model, weights_path)
        print("Locked and loaded.")

        self.lock.release()

        return model

    def start_webserver(self, host, port):
        import cherrypy
        import numpy

        class WebServer(object):
            def __init__(self, lock, job_model, model):
                self.job_model = job_model
                self.model = model
                self.lock = lock

            @cherrypy.expose
            def predict(self, path=None, paths=None, uploads=None, inputs=None):
                self.lock.acquire()
                result = {'times': {}}

                try:
                    start = time.time()

                    if not path and not paths and not uploads and not inputs:
                        return json.dumps({'error': 'not_input_given'})

                    if path and not paths and not uploads and not inputs:
                        paths = [path]

                    encoded_inputs = []
                    if paths:
                        if not isinstance(paths, list):
                            paths = [paths]

                        for idx, file_path in enumerate(paths):
                            encoded_inputs.append(self.job_model.convert_file_to_input_node(file_path, self.job_model.get_input_node(idx)))

                    if inputs:
                        if not isinstance(inputs, list):
                            inputs = [inputs]

                        for idx, input in enumerate(inputs):
                            encoded_inputs.append(self.job_model.encode_input_to_input_node(input, self.model.input_layers[idx]))

                    if uploads:
                        if not isinstance(uploads, list):
                            uploads = [uploads]

                        for idx, upload in enumerate(uploads):
                            encoded_inputs.append(self.job_model.convert_file_to_input_node(upload.filename, self.job_model.get_input_node(idx)))

                    result['times']['prepare_fetch_input'] = time.time() - start

                    start = time.time()
                    prediction = self.job_model.predict(self.model, numpy.array(encoded_inputs))
                    result['times']['prediction'] = time.time() - start

                    self.lock.release()
                except Exception as e:
                    self.lock.release()
                    return json.dumps({'error': type(e).__name__, 'message': e.message})

                result['prediction'] = prediction

                return json.dumps(result)

        cherrypy.config.update({
            'server.socket_host': host,
            'server.socket_port': port,
            'server.thread_pool': 1,
        })

        print("Starting server ... Use http://127.0.0.1:8000/predict?paths=path_to_image.jpg or upload files named 'uploads[0]' to http://127.0.0.1:8000/predict.")
        cherrypy.quickstart(WebServer(self.lock, self.job_model, self.model))
