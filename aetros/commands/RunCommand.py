from __future__ import absolute_import
from __future__ import print_function
import argparse

import sys

from aetros.backend import JobBackend
from aetros import api
from aetros.utils import read_config, human_size

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
        parser.add_argument('command', nargs='?', help="The command to run. Default read in .aetros.yml")
        parser.add_argument('-i', '--image', help="Which Docker image to use for the command. Default read in .aetros.yml. If not specified, command is executed on the host.")
        parser.add_argument('-s', '--server', help="On which AETROS server the command should be executed. Default read in .aetros.yml")
        parser.add_argument('-m', '--model', help="Under which model this job should be listed. Default read in .aetros.yml")
        parser.add_argument('-l', '--local', action='store_true', help="Start the job immediately on the current machine.")
        parser.add_argument('-c', '--config', help="Default .aetros.yml in current working directory.")

        parsed_args = parser.parse_args(args)

        config = read_config(parsed_args.config or '.aetros.yml')
        model_name = parsed_args.model

        if 'command' not in config and not parsed_args.command:
            self.logger.error('No "command" given in .aetros.yml or as argument.')
            sys.exit(1)

        if not model_name and 'model' not in config:
            raise Exception('No model name given. Specify in .aetros.yml or --model=model/name')
        if not model_name and 'model' in config:
            model_name = config['model']

        job = JobBackend(parsed_args.model, self.logger, parsed_args.config or '.aetros.yml')

        files_added, size_added = job.add_files()
        print("%d files added (%s)" % (files_added, human_size(size_added, 2)))

        create_info = api.create_job_info(model_name)
        config.update(create_info['config'])
        if parsed_args.command:
            create_info['config']['command'] = parsed_args.command

        if parsed_args.image:
            create_info['config']['image'] = parsed_args.image

        if parsed_args.server:
            create_info['config']['server'] = parsed_args.server

        if parsed_args.local:
            create_info['config']['server'] = 'local'

        create_info['config']['sourcesAttached'] = True
        create_info['config'] = config

        job.create(create_info=create_info)
        job.git.push()

        print("Job %s/%s created." % (job.model_name, job.job_id))
        print("Open http://%s/model/%s/job/%s to monitor it." % (job.host, job.model_name, job.job_id))
