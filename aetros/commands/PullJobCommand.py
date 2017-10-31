from __future__ import absolute_import
from __future__ import print_function
import argparse
import subprocess

import sys

import os

from aetros.utils import read_home_config, setup_git_ssh


class PullJobCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' pull-job')
        parser.add_argument('id', nargs='?', help="Model name like peter/mnist/ef8009d83a9892968097cec05b9467c685d45453")

        parsed_args = parser.parse_args(args)

        if not parsed_args.id:
            parser.print_help()
            sys.exit(1)

        config = read_home_config()
        model = parsed_args.id[0:parsed_args.id.rindex('/')]
        ref = 'refs/aetros/job/' + parsed_args.id[parsed_args.id.rindex('/')+1:]

        git_dir = os.path.normpath(config['storage_dir'] + '/' + model + '.git')

        if not os.path.isdir(git_dir):
            self.logger.error("Git repository for model %s in %s not found." % (parsed_args.id, git_dir))
            self.logger.error("Are you in the correct directory?")

        print('Pull ' + ref + ' into ' + git_dir)
        setup_git_ssh(config)
        subprocess.call([config['git'], '--bare', '--git-dir', git_dir, 'fetch', 'origin', ref+':'+ref])