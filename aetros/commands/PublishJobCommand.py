from __future__ import absolute_import
from __future__ import print_function
import argparse
import subprocess

import sys

import os

from aetros import api
from aetros.utils import read_config


class PublishJobCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                description="You can provide json in stdin to issue a POST call", prog=aetros.const.__prog__ + ' run')
        parser.add_argument('model', nargs='?', help="Model name like peter/mnist")
        parser.add_argument('ref', nargs='?', help="Git ref name like refs/aetros/job/ef8009d83a9892968097cec05b9467c685d45453/head")

        parsed_args = parser.parse_args(args)

        if not parsed_args.model or not parsed_args.ref:
            parser.print_help()
            sys.exit(1)

        git_dir = '.aetros/' + parsed_args.model + '.git'

        if not os.path.isdir(git_dir):
            self.logger.error("Git repository for model %s in %s not found." % (parsed_args.model, git_dir))
            self.logger.error("Are you in the correct directory?")

        subprocess.call(['git', '--bare', '--git-dir', git_dir, 'push', 'origin', '-f', parsed_args.ref])