from __future__ import absolute_import
from __future__ import print_function
import argparse
import os
import sys
import time
import yaml

from aetros.backend import BackendClient, EventListener, ApiClient, JobBackend
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
        parser.add_argument('command', help="The command to run. Default read in .aetros.yml")
        parser.add_argument('--image', help="Which Docker image to use for the command. Default read in .aetros.yml. If not specified, command is executed on the host.")
        parser.add_argument('--server', help="On which AETROS server the command should be executed. Default read in .aetros.yml")
        parser.add_argument('--model', help="Under which model this job should be listed. Default read in .aetros.yml")
        parser.add_argument('--host', help="Default trainer.aetros.com. Default read in ~/.aetros.yml.")
        parser.add_argument('--config', help="Default .aetros.yml in current working directory.")

        parsed_args = parser.parse_args(args)

        config = read_config(parsed_args.config if parsed_args.config else '.aetros.yml')
        model_name = parsed_args.model

        if not model_name and 'model' not in config:
            raise Exception('No model name given. Specify in .aetros.yml or --model=model/name')
        if not model_name and 'model' in config:
            model_name = config['model']

        # create git root commit with
        # aetros/job/job.json
        # with server requirement specification
        # git push
        # wait for job to run
        # print logs of job

        job = JobBackend(parsed_args.model, self.logger)

        files_added, size_added = job.add_files()
        self.logger.info("%d files added (%s)" % (files_added, human_size(size_added, 2)))

        create_info = api.create_job_info(model_name)
        if parsed_args.command:
            create_info['config']['command'] = parsed_args.command

        if parsed_args.image:
            create_info['config']['image'] = parsed_args.image

        create_info['config']['sourceGitDisabled'] = True
        print(create_info)
        job.create(create_info=create_info, server=parsed_args.server or 'local')
        job.git.push()

        self.logger.info("Job %s/%s started." % (job.model_name, job.job_id))
        self.logger.info("Open http://%s/model/%s/job/%s to monitor it." % (job.host, job.model_name, job.job_id))


    #     event_listener = EventListener()
    #
    #     model_id = None
    #
    #     if config['model']:
    #         model_id = config['model']
    #     if parsed_args.model:
    #         model_id = parsed_args.model
    #
    #     event_listener.on('registration', self.on_registration)
    #     event_listener.on('disconnect', self.on_disconnect)
    #     event_listener.on('stop', self.on_stop)
    #
    #     self.client = RunClient(config, event_listener)
    #     self.client.configure(model_id)
    #     self.client.start()
    #
    #     while self.active:
    #         if self.registered:
    #             pass
    #
    #         time.sleep(1)
    #
    # def on_registration(self, params):
    #     self.registered = True
    #
    # def on_disconnect(self, params):
    #     self.registered = False
    #
    # def on_stop(self, params):
    #     self.active = False
