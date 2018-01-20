from __future__ import absolute_import
from __future__ import print_function
import argparse

import os

import sys
import ruamel

from aetros import api
from aetros.utils import read_home_config


class InitCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
            prog=aetros.const.__prog__ + ' run')
        parser.add_argument('name', nargs='?', help="Model name")
        parser.add_argument('--space', '-s', help="Create the model in given space. If space does not exist, create it.")
        parser.add_argument('--private', action='store_true', help="Make the model private. Example: aetros init my-model --private")
        parser.add_argument('--overwrite', '-o', action='store_true', help="Overwrite already existing configuration.")

        home_config = read_home_config()
        parsed_args = parser.parse_args(args)
        if not parsed_args.name:
            parser.print_help()
            sys.exit(1)

        yaml = ruamel.yaml.YAML()
        config = {}

        if os.path.exists('aetros.yml'):
            with open('aetros.yml', 'r') as f:
                config = yaml.load(f)

            if isinstance(config, dict) and 'model' in config and not parsed_args.overwrite:
                print("failed: aetros.yml already exists with a linked model to " + config['model']+ '. Use -o to overwrite.')
                sys.exit(1)

        if not parsed_args.private:
            print("Warning: creating public model. Use --private to create private models.")

        try:
            name = api.create_model(parsed_args.name or (os.path.basename(os.getcwd())), parsed_args.space, parsed_args.private)
        except api.ApiError as e:
            if e.error != 'already_exists':
                raise e
            print("Notice: Model already exists remotely.")
            name = parsed_args.name

        config['model'] = name

        with open('aetros.yml', 'w+') as f:
            yaml.dump(config, f)

        print("aetros.yml created linked with model " + name + ' in ' + os.getcwd())
        print("Open AETROS Trainer to see the model at https://" + home_config['host'] + '/model/' + name)
