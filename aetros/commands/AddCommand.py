from __future__ import absolute_import
from __future__ import print_function
import argparse
import subprocess

import sys

import os

from aetros.git import Git
from aetros.utils import read_config, unpack_simple_job_id

class AddCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' add')
        parser.add_argument('id', nargs='?', help="Model name like peter/mnist/ef8009d83a9892968097cec05b9467c685d45453")
        parser.add_argument('local_path', nargs='?', help="File path to be added to the job head commit history")
        parser.add_argument('git_path', nargs='?', help="Path under which the file will be stored.")
        parser.add_argument('-m', help="Commit message")

        parsed_args = parser.parse_args(args)

        if not parsed_args.id or not parsed_args.local_path:
            parser.print_help()
            sys.exit(1)

        config = read_config()

        [model, job_id] = unpack_simple_job_id(parsed_args.id)
        ref = 'refs/aetros/job/' + job_id

        git_dir = os.path.normpath(config['storage_dir'] + '/' + model + '.git')

        if not os.path.isdir(git_dir):
            self.logger.error("Git repository for model %s in %s not found." % (parsed_args.id, git_dir))
            self.logger.error("Are you in the correct directory?")

        git = Git(self.logger, None, config, model)
        git.job_id = job_id
        try:
            git.read_tree(ref)

            with open(parsed_args.local_path, 'rb') as f:
                id = git.commit_file(parsed_args.m or 'Added file', parsed_args.git_path or parsed_args.local_path, f.read())

                self.logger.info('Successfully committed to ' + id + '.')
                self.logger.info('Run "aetros push-job ' + parsed_args.id + '" to upload changes to AETROS.')
        finally:
            git.clean_up()