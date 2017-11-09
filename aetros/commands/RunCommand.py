from __future__ import absolute_import
from __future__ import print_function
import argparse

import sys

import os

import aetros.utils.git
from aetros.logger import GeneralLogger

from aetros.starter import start

from aetros.backend import JobBackend
from aetros import api
from aetros.utils import read_config, human_size, lose_parameters_to_full, extract_parameters, stop_time


class RunCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' run')
        parser.add_argument('command', nargs='?', help="The command to run. Default read in aetros.yml")
        parser.add_argument('-i', '--image', help="Which Docker image to use for the command. Default read in aetros.yml. If not specified, command is executed on the host.")
        parser.add_argument('-s', '--server', action='append', help="Limits the server pool to this server. Default not limitation or read in aetros.yml. Multiple --server allowed.")
        parser.add_argument('-m', '--model', help="Under which model this job should be listed. Default read in aetros.yml")
        parser.add_argument('-l', '--local', action='store_true', help="Start the job immediately on the current machine.")
        parser.add_argument('-c', '--config', help="Default aetros.yml in current working directory.")
        parser.add_argument('--priority', help="Increases or decreases priority. Default is 0.")

        parser.add_argument('--cpu', help="How many CPU cores should be assigned to job. Docker only.")
        parser.add_argument('--memory', help="How much memory should be assigned to job. Docker only.")
        parser.add_argument('--gpu', help="How many GPU cards should be assigned to job. Docker only.")
        parser.add_argument('--gpu_memory', help="Memory requirement for the GPU. Docker only.")

        parser.add_argument('--max-time', help="Limit execution time in seconds. Sends SIGINT to the process group when reached.")
        parser.add_argument('--max-epochs', help="Limit execution epochs. Sends SIGINT to the process group when reached.")

        parser.add_argument('--gpu-device', action='append', help="Which device id should be mapped into the NVIDIA docker container.")

        parser.add_argument('--volume', '-v', action='append', help="Volume into docker")
        parser.add_argument('-e', action='append', help="Sets additional environment variables. '-e name=value' to set value, or '-e name' to read from current env")

        parser.add_argument('-p', '--param', action='append', help="Sets a hyperparameter, example '--param name=value'. Multiple --param allowed.")

        parsed_args = parser.parse_args(args)

        config = read_config(parsed_args.config or 'aetros.yml')

        env = {}
        if parsed_args.e:
            for item in parsed_args.e:
                if '=' in item:
                    k, v = item.split('=')
                else:
                    k = item
                    v = os.getenv(k)
                env[k] = v

        if 'command' not in config and not parsed_args.command:
            self.logger.error('No "command" given in aetros.yml or as argument.')
            sys.exit(1)

        job = JobBackend(parsed_args.model, self.logger, parsed_args.config or 'aetros.yml')
        ignore = []
        if 'ignore' in config:
            ignore = config['ignore']
        job.job = {'config': {'ignore': ignore}}

        files_added, size_added = job.add_files()

        print("%d files added (%s)" % (files_added, human_size(size_added, 2)))

        create_info = {
            'type': 'custom',
            'config': config
        }

        incoming_hyperparameter = {}
        if parsed_args.param:
            for param in parsed_args.param:
                if '=' not in param:
                    raise Exception('--param ' + param + ' does not contain a `=`. Please use "--param name=value"')

                name, value = param.split('=')
                incoming_hyperparameter[name] = value

        # first transform simple format in the full definition with parameter types
        # (string, number, group, choice_group, etc)
        full_hyperparameters = lose_parameters_to_full(config['parameters'])

        # now extract hyperparameters from full definition, and overwrite stuff using
        # incoming_hyperparameter if available
        hyperparameter = extract_parameters(full_hyperparameters, incoming_hyperparameter)

        create_info['config']['parameters'] = hyperparameter

        if parsed_args.max_epochs:
            create_info['config']['maxEpochs'] = int(parsed_args.max_epochs)

        create_info['config']['priority'] = 0
        if parsed_args.priority:
            create_info['config']['priority'] = float(parsed_args.priority)

        if parsed_args.max_time:
            create_info['config']['maxTime'] = float(parsed_args.max_time)

        if parsed_args.command:
            create_info['config']['command'] = parsed_args.command

        if parsed_args.image:

            # reset install options, since we can't make sure if the base image still fits
            if 'image' in config and config['image'] and config['image'] != parsed_args.image:
                create_info['config']['install'] = None

            # reset dockerfile, since we specified manually an image
            create_info['config']['dockerfile'] = None
            create_info['config']['image'] = parsed_args.image

        if parsed_args.server:
            create_info['config']['servers'] = []
            for name in parsed_args.server:
                create_info['config']['servers'].append(name)

        if 'resources' not in create_info['config']:
            create_info['config']['resources'] = {}

        if parsed_args.cpu or parsed_args.memory or parsed_args.gpu is not None or parsed_args.gpu_memory:
            if parsed_args.cpu: create_info['config']['resources']['cpu'] = float(parsed_args.cpu)
            if parsed_args.memory: create_info['config']['resources']['memory'] = float(parsed_args.memory)
            if parsed_args.gpu is not None: create_info['config']['resources']['gpu'] = float(parsed_args.gpu)
            if parsed_args.gpu_memory: create_info['config']['resources']['gpu_memory'] = float(parsed_args.gpu_memory)

        if parsed_args.local:
            # usually, the aetros server would assign resources at job root level from the assigned server
            # but since it's started locally, we just use the requested one. User should know what they do.
            # start.py will use 'config' stuff anyone for docker limitation, so we should make sure it is
            # being displayed.

            if 'image' in create_info['config'] and create_info['config']:
                resources = create_info['config']['resources']
                create_info['resources_assigned'] = {'cpus': 1, 'memory': 1, 'gpus': []}

                if 'gpu' in resources and resources['gpu'] > 0:
                    create_info['resources_assigned']['gpus'] = [1] * resources['gpu']
                if 'cpu' in resources:
                    create_info['resources_assigned']['cpus'] = resources['cpu']
                if 'memory' in resources:
                    create_info['resources_assigned']['memory'] = resources['memory']
            else:
                # since this runs on the host, extract machine hardware and put int resources_assigned
                # so we see it at the job.
                pass

        if parsed_args.local:
            create_info['server'] = 'local'

        create_info['config']['sourcesAttached'] = True

        if aetros.utils.git.get_current_commit_hash():
            create_info['origin_git_source'] = {
                'origin': aetros.utils.git.get_current_remote_url(),
                'author': aetros.utils.git.get_current_commit_author(),
                'message': aetros.utils.git.get_current_commit_message(),
                'branch': aetros.utils.git.get_current_branch(),
                'commit': aetros.utils.git.get_current_commit_hash(),
            }

        job.create(create_info=create_info, server=None)

        print("Job %s/%s created." % (job.model_name, job.job_id))

        if parsed_args.local:
            start(self.logger, job.model_name + '/' + job.job_id, fetch=False, env=env, volumes=parsed_args.volume, gpu_devices=parsed_args.gpu_device)
        else:
            if parsed_args.volume:
                print("Can not use volume with jobs on the cluster. Use datasets instead.")
                sys.exit(1)

            #todo, make it visible
            job.git.push()
            print("Open http://%s/model/%s/job/%s to monitor it." % (job.host, job.model_name, job.job_id))
