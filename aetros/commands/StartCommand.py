from __future__ import absolute_import
import argparse
import sys

from aetros import api
from aetros.utils import read_home_config
import aetros.const


class StartCommand:
    def __init__(self, logger):
        self.logger = logger

    def main(self, args):
        from aetros.starter import start
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' start')
        parser.add_argument('name', help='the model name, e.g. aetros/mnist-network to start new job, or job id, e.g. user/modelname/0db75a64acb74c27bd72c22e359de7a4c44a20e5 to start a pre-created job.')

        parser.add_argument('-i', '--image', help="Which Docker image to use for the command. Default read in aetros.yml. If not specified, command is executed on the host.")
        parser.add_argument('-l', '--local', action='store_true', help="Start the job immediately on the current machine.")
        parser.add_argument('-s', '--server', action='append', help="Limits the server pool to this server. Default not limitation or read in aetros.yml. Multiple --server allowed.")
        parser.add_argument('-b', '--branch', help="This overwrites the Git branch used when new job should be started.")
        parser.add_argument('-c', '--config', help="Default /aetros.yml in Git root.")
        parser.add_argument('--priority', help="Increases or decreases priority. Default is 0.")

        parser.add_argument('--cpu', help="How many CPU cores should be assigned to job. Docker only.")
        parser.add_argument('--memory', help="How much memory should be assigned to job. Docker only.")
        parser.add_argument('--gpu', help="How many GPU cards should be assigned to job. Docker only.")
        parser.add_argument('--gpu_memory', help="Memory requirement for the GPU. Docker only.")

        parser.add_argument('--rebuild-image', action='store_true', help="Makes sure the Docker image is re-built without cache.")

        parser.add_argument('--gpu-device', action='append', help="Which GPU device id should be mapped into the Docker container. Only with --local.")

        parser.add_argument('--max-time', help="Limit execution time in seconds. Sends SIGINT to the process group when reached.")
        parser.add_argument('--max-epochs', help="Limit execution epochs. Sends SIGINT to the process group when reached.")

        parser.add_argument('--insights', action='store_true', help="activates insights. Only for simple models.")
        parser.add_argument('--dataset', help="Dataset id when model has placeholders. Only for simple models with placeholders as input/output.")

        parser.add_argument('-p', '--param', action='append', help="Sets a hyperparameter, example '--param name=value'. Multiple --param allowed.")

        parsed_args = parser.parse_args(args)

        if not parsed_args.name:
            print("fatal: no model defined. 'aetros start user/model-name'.")
            sys.exit(2)

        if parsed_args.name and parsed_args.name.count('/') > 1:
            # start a concrete job, used by server command
            gpu_devices = []
            if parsed_args.gpu_device:
                gpu_devices = [int(x) for x in parsed_args.gpu_device]

            start(self.logger, parsed_args.name, cpus=int(parsed_args.cpu), memory=int(parsed_args.memory),
                gpu_devices=gpu_devices)
            return

        home_config = read_home_config()
        model_name = parsed_args.name

        # create a new job
        hyperparameter = {}
        if parsed_args.param:
            for param in parsed_args.param:
                if '=' not in param:
                    raise Exception('--param ' + param + ' does not contain a `=`. Please use "--param name=value"')

                name, value = param.split('=')
                hyperparameter[name] = value

        job_config = {'insights': parsed_args.insights}

        if parsed_args.image:
            job_config['image'] = parsed_args.image

        if parsed_args.branch:
            job_config['sourceGitTree'] = parsed_args.branch

        if parsed_args.max_epochs:
            job_config['maxEpochs'] = int(parsed_args.max_epochs)

        if parsed_args.max_time:
            job_config['maxTime'] = float(parsed_args.max_time)

        job_config['priority'] = 0
        if parsed_args.priority:
            job_config['priority'] = float(parsed_args.priority)

        if parsed_args.rebuild_image:
            job_config['config']['rebuild_image'] = True

        if parsed_args.server:
            job_config['servers'] = []
            for name in parsed_args.server:
                job_config['servers'].append(name)

        job_config['resources'] = {}

        if parsed_args.cpu:
            job_config['resources']['cpu'] = int(parsed_args.cpu)

        if parsed_args.memory:
            job_config['resources']['memory'] = int(parsed_args.memory)

        if parsed_args.gpu:
            job_config['resources']['gpu'] = int(parsed_args.gpu)

        if parsed_args.gpu_memory:
            job_config['resources']['gpu_memory'] = int(parsed_args.gpu_memory)

        config_path = parsed_args.config or 'aetros.yml'

        try:
            self.logger.debug("Create job ...")
            created = api.create_job(model_name, config_path, parsed_args.local, hyperparameter, parsed_args.dataset, config=job_config)
        except api.ApiError as e:
            if 'Connection refused' in e.error:
                self.logger.error("You are offline")

            raise

        self.logger.info("Job %s/%s created." % (model_name, created['id']))

        if parsed_args.local:
            start(self.logger, model_name + '/' + created['id'], gpu_devices=parsed_args.gpu_device)
        else:
            print("Open http://%s/model/%s/job/%s to monitor it." % (home_config['host'], model_name, created['id']))
