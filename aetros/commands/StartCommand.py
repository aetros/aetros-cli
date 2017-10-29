from __future__ import absolute_import
import argparse
import sys

from aetros import api

from aetros.utils import read_config, unpack_full_job_id, read_home_config

import aetros.const
import os

class StartCommand:
    def __init__(self, logger):
        self.logger = logger

    def main(self, args):
        from aetros.starter import start
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' start')
        parser.add_argument('name', nargs='?', help='the model name, e.g. aetros/mnist-network to start new job, or job id, e.g. user/modelname/0db75a64acb74c27bd72c22e359de7a4c44a20e5 to start a pre-created job.')

        parser.add_argument('-i', '--image', help="Which Docker image to use for the command. Default read in aetros.yml. If not specified, command is executed on the host.")
        parser.add_argument('-l', '--local', action='store_true', help="Start the job immediately on the current machine.")
        parser.add_argument('-s', '--server', help="Limits the server pool to this server. Default not limitation or read in aetros.yml.")
        parser.add_argument('-b', '--branch', help="This overwrites the Git branch used when new job should be started.")

        parser.add_argument('--insights', action='store_true', help="activates insights. Only for simple models.")
        parser.add_argument('--dataset', help="Dataset id when model has placeholders. Only for simple models with placeholders as input/output.")

        parser.add_argument('--param', action='append', help="Sets a hyperparameter, example '--param name=value'. Multiple --param allowed.")

        parsed_args = parser.parse_args(args)

        home_config = read_home_config()

        hyperparameter = {}
        if parsed_args.param:
            for param in parsed_args.param:
                if '=' not in param:
                    raise Exception('--param ' + param + ' does not contain a `=`. Please use "--param name=value"')

                name, value = param.split('=')
                hyperparameter[name] = value

        job_config = {'insights': parsed_args.insights}

        if parsed_args.local:
            job_config['server'] = 'local'

        if parsed_args.image:
            job_config['image'] = parsed_args.image

        if parsed_args.branch:
            job_config['sourceGitTree'] = parsed_args.branch

        model_name = parsed_args.name

        if model_name.count('/') == 1:
            try:
                self.logger.debug("Create job ...")
                created = api.create_job(model_name, hyperparameter, parsed_args.dataset, config=job_config)
            except api.ApiError as e:
                if 'Connection refused' in e.reason:
                    self.logger.error("You are offline")

                raise

            print("Job %s/%s created." % (model_name, created['id']))

            if parsed_args.local:
                start(self.logger, model_name + '/' + created['id'])
            else:
                print("Open http://%s/model/%s/job/%s to monitor it." % (home_config['host'], model_name, created['id']))

        else:
            start(self.logger, model_name)
