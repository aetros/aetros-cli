from __future__ import absolute_import
from __future__ import print_function
import argparse
import subprocess

import sys

import os

from aetros.utils import read_home_config, setup_git_ssh, read_config, find_config, git_has_local_job


class JobPushCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' job-push')
        parser.add_argument('id', help="Short or long job id, like ef8009d83a9892968097cec05b9467c685d45453")
        parser.add_argument('--model', help="Model name like peter/mnist. Per default from current directory")
        parser.add_argument('-c', '--config', help="Default aetros.yml in current working directory.")

        parsed_args = parser.parse_args(args)

        if not parsed_args.id:
            parser.print_help()
            sys.exit(1)

        home_config = read_home_config()
        config = find_config(parsed_args.config)
        model = parsed_args.model if parsed_args.model else config['model']

        if not model:
            print("No model defined. Use --model or switch into a directory where you executed 'aetros init model-name'.")
            sys.exit(2)

        full_id = git_has_local_job(home_config, model, parsed_args.id)
        if not full_id:
            print("Error: Job not found on local.")
            sys.exit(1)

        ref = 'refs/aetros/job/' + full_id
        git_dir = os.path.normpath(home_config['storage_dir'] + '/' + model + '.git')

        if not os.path.isdir(git_dir):
            self.logger.error("Git repository for model %s in %s not found." % (full_id, git_dir))
            self.logger.error("You seem not to have any job created on this machine for model " + model)
            sys.exit(1)

        print('Push job %s of %s' % (full_id, model))
        setup_git_ssh(home_config)
        subprocess.call([home_config['git'], '--bare', '--git-dir', git_dir, 'push', 'origin', ref])