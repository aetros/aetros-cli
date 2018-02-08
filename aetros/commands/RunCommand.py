# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import argparse

import sys
import os
from math import ceil

import psutil
import six
from cpuinfo import cpuinfo

import aetros.utils.git
from aetros.cuda_gpu import get_ordered_devices, CudaNotImplementedException
from aetros.starter import start_command

from aetros.backend import JobBackend
from aetros.utils import human_size, lose_parameters_to_full, extract_parameters, find_config, loading_text, \
    read_home_config, ensure_docker_installed, docker_call


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
        parser.add_argument('command', nargs='?', help="The command to run. Default read in configuration file")
        parser.add_argument('-i', '--image', help="Which Docker image to use for the command. Default read in configuration file. If not specified, command is executed on the host.")
        parser.add_argument('--no-image', action='store_true', help="Forces not to use docker, even when image is defined in the configuration file.")

        parser.add_argument('-s', '--server', action='append', help="Limits the server pool to this server. Default not limitation or read in configuration file. Multiple --server allowed.")
        parser.add_argument('-m', '--model', help="Under which model this job should be listed. Default read in configuration file")
        parser.add_argument('-l', '--local', action='store_true', help="Start the job immediately on the current machine.")
        parser.add_argument('-c', '--config', help="Default aetros.yml in current working directory.")
        parser.add_argument('--priority', help="Increases or decreases priority. Default is 0.")

        parser.add_argument('--cpu', help="How many CPU cores should be assigned to job. Docker only.")
        parser.add_argument('--memory', help="How much memory should be assigned to job. Docker only.")
        parser.add_argument('--gpu', help="How many GPU cards should be assigned to job. Docker only.")
        parser.add_argument('--gpu_memory', help="Memory requirement for the GPU. Docker only.")

        parser.add_argument('--offline', '-o', action='store_true', help="Whether the execution should happen offline.")

        parser.add_argument('--rebuild-image', action='store_true', help="Makes sure the Docker image is re-built without cache.")

        parser.add_argument('--max-time', help="Limit execution time in seconds. Sends SIGINT to the process group when reached.")
        parser.add_argument('--max-epochs', help="Limit execution epochs. Sends SIGINT to the process group when reached.")

        parser.add_argument('--gpu-device', action='append', help="Which device id should be mapped into the NVIDIA docker container. Only when --local")

        parser.add_argument('--volume', '-v', action='append', help="Volume into docker. Only when --local")
        parser.add_argument('-e', action='append', help="Sets additional environment variables. '-e name=value' to set value, or '-e name' to read from current env")

        parser.add_argument('-p', '--param', action='append', help="Sets a hyperparameter, example '--param name=value'. Multiple --param allowed.")

        parsed_args = parser.parse_args(args)

        if parsed_args.config and not os.path.exists(parsed_args.config):
            self.logger.error("fatal: file %s does not exist." % (parsed_args.config,))
            sys.exit(2)

        config = find_config(parsed_args.config)
        home_config = read_home_config()

        if config['model'] and not parsed_args.model:
            parsed_args.model = config['model']

        if not parsed_args.model:
            print("fatal: no model defined. Use --model or switch into a directory where you executed 'aetros init model-name'.")
            sys.exit(2)

        if not parsed_args.local and parsed_args.volume:
            print("fatal: can not use volume with jobs on the cluster. Use datasets instead.")
            sys.exit(1)

        if parsed_args.local and parsed_args.priority:
            print("fatal: the priority can only be set for jobs in the cluster.")
            sys.exit(1)

        if config['image']:
            ensure_docker_installed(self.logger)

        env = {}
        if parsed_args.e:
            for item in parsed_args.e:
                if '=' in item:
                    k, v = item.split('=')
                else:
                    k = item
                    v = os.getenv(k)
                env[k] = v

        if ('command' not in config or not config['command']) and not parsed_args.command:
            self.logger.error('No command given. Define the command in aetros.yml or use command argument.')
            sys.exit(1)

        job_backend = JobBackend(parsed_args.model, self.logger)

        ignore = []
        if 'ignore' in config:
            ignore = config['ignore']
        job_backend.job = {'config': {'ignore': ignore}}

        adding_files = loading_text("- Adding job files to index ... ")
        files_added, size_added = job_backend.add_files(config['root'], report=False)
        adding_files("done with %d file%s added (%s)."
                     % (files_added, 's' if files_added != 1 else '', human_size(size_added, 2)))

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

        if parsed_args.rebuild_image:
            create_info['config']['rebuild_image'] = True

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

        if parsed_args.no_image:
            create_info['config']['image'] = None

        if parsed_args.server:
            create_info['config']['servers'] = []
            for name in parsed_args.server:
                create_info['config']['servers'].append(name)

        create_info['config']['resources'] = create_info['config'].get('resources', {})
        resources = create_info['config']['resources']

        default_cpu_and_memory = 1 if create_info['config']['image'] else 0
        resources['cpu'] = int(parsed_args.cpu or resources.get('cpu', default_cpu_and_memory))
        resources['memory'] = int(parsed_args.memory or resources.get('memory', default_cpu_and_memory))
        resources['gpu'] = int(parsed_args.gpu or resources.get('gpu', 0))
        resources['gpu_memory'] = int(parsed_args.gpu_memory or resources.get('gpu_memory', 0))

        if parsed_args.local:
            create_info['server'] = 'local'

            # make sure we do not limit the resources to something that is not available on the local machine
            warning = []
            cpu = cpuinfo.get_cpu_info()
            mem = psutil.virtual_memory().total
            gpu = 0
            try:
                gpu = len(get_ordered_devices())
            except CudaNotImplementedException: pass

            if not create_info['config']['image'] and not all([x == 0 for x in six.itervalues(resources)]):
                self.logger.warning("! No Docker virtualization since no `image` defined, resources limitation ignored.")

            if create_info['config']['image'] and resources['gpu'] > 0:
                if not (sys.platform == "linux" or sys.platform == "linux2"):
                    self.logger.warning("! Your operating system does not support GPU allocation for "
                                        "Docker virtualization. "
                                        "NVIDIA-Docker2 is only supported on Linux.")

            local_max_resources = {'cpu': cpu['count'], 'memory': ceil(mem / 1024 / 1024 / 1024), 'gpu': gpu}

            if create_info['config']['image']:
                # read max hardware within Docker
                out = docker_call(['run', 'alpine', 'sh', '-c', 'nproc && cat /proc/meminfo | grep MemTotal'])
                cpus, memory = out.decode('utf-8').strip().split('\n')
                local_max_resources['cpu'] = int(cpus)

                memory = memory.replace('MemTotal:', '').replace('kB', '').strip()
                local_max_resources['memory'] = ceil(int(memory) / 1024 / 1024)

            if local_max_resources['cpu'] < resources['cpu']:
                warning.append('CPU cores %d -> %d' % (resources['cpu'], local_max_resources['cpu']))
                resources['cpu'] = local_max_resources['cpu']

            if local_max_resources['memory'] < resources['memory']:
                warning.append('memory %dGB -> %dGB' % (resources['memory'], local_max_resources['memory']))
                resources['memory'] = local_max_resources['memory']

            if local_max_resources['gpu'] < resources['gpu']:
                warning.append('GPU cards %d -> %d' % (resources['gpu'], local_max_resources['gpu']))
                resources['gpu'] = local_max_resources['gpu']

            if warning:
                self.logger.warning("! Resources downgrade due to missing hardware: %s." % (', '.join(warning),))

        if parsed_args.config and not create_info['config']['configPath']:
            create_info['config']['configPath'] = parsed_args.config

        create_info['config']['sourcesAttached'] = True

        creating_git_job = loading_text("- Create job in local Git ... ")
        if aetros.utils.git.get_current_commit_hash():
            create_info['origin_git_source'] = {
                'origin': aetros.utils.git.get_current_remote_url(),
                'author': aetros.utils.git.get_current_commit_author(),
                'message': aetros.utils.git.get_current_commit_message(),
                'branch': aetros.utils.git.get_current_branch(),
                'commit': aetros.utils.git.get_current_commit_hash(),
            }

        job_backend.create(create_info=create_info, server=None)
        creating_git_job("created %s in %s." % (job_backend.job_id[0:9], job_backend.model_name))

        summary = "➤ Summary: Job running "
        if parsed_args.local:
            summary += 'locally'
        else:
            summary += 'on the cluster'

        if create_info['config']['image']:
            summary += ' in Docker using image %s with %d CPU cores, %d memory and %d GPUs.' \
                       % (create_info['config']['image'], resources['cpu'], resources['memory'], resources['gpu'])
        else:
            summary += ' on host using all available resources.'

        print(summary)

        # tasks = []
        #
        # if 'tasks' in config:
        #     for name, task_config in six.iteritems(config['tasks']):
        #         replica = 1
        #         if 'replica' in task_config:
        #             replica = int(task_config['replica'])
        #         for index in range(0, replica):
        #             tasks.append(job_backend.create_task(job_id, task_config, name, index))

        if parsed_args.offline:
            if not parsed_args.local:
                self.logger.warning("Can not create a remote job in offline mode.")
                sys.exit(1)

            self.logger.warning("Execution started offline.")
        else:
            adding_files = loading_text("- Connecting to "+home_config['host']+" ... ")
            if job_backend.connect():
                adding_files("connected.")
            else:
                parsed_args.offline = True
                adding_files("failed. Continue in offline mode.")

        if not parsed_args.offline:
            sys.stdout.write("- Uploading job data ... ")
            job_backend.git.push()
            job_backend.client.wait_until_queue_empty(['files'], clear_end=False)

            sys.stdout.write(" done.\n")

            link = "%smodel/%s/job/%s" % (home_config['url'], job_backend.model_name, job_backend.job_id)
            sys.__stdout__.write(u"➤ Monitor job at %s\n" % (link))

        if parsed_args.local:
            job_backend.start(collect_system=False, offline=parsed_args.offline, push=False)

            if not parsed_args.offline:
                job_backend.git.start_push_sync()

            cpus = create_info['config']['resources']['cpu']
            memory = create_info['config']['resources']['memory']

            if not parsed_args.gpu_device and create_info['config']['resources']['gpu'] > 0:
                # if requested 2 GPUs and we have 3 GPUs with id [0,1,2], gpus should be [0,1]
                parsed_args.gpu_device = []
                for i in range(0, create_info['config']['resources']['gpu']):
                    parsed_args.gpu_device.append(i)

            start_command(self.logger, job_backend, env, parsed_args.volume, cpus=cpus, memory=memory, gpu_devices=parsed_args.gpu_device,
                offline=parsed_args.offline)
