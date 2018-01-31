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
            prog=aetros.const.__prog__ + ' init')
        parser.add_argument('name', help="Model name")
        parser.add_argument('directory', nargs='?', help="Directory, default in current.")
        parser.add_argument('--organisation', '-o', help="Create the model in the organisation instead of the user account.")
        parser.add_argument('--space', '-s', help="Create the model in given space. If space does not exist, create it.")
        parser.add_argument('--private', action='store_true', help="Make the model private. Example: aetros init my-model --private")
        parser.add_argument('--force', '-f', action='store_true', help="Force overwriting of already existing configuration file.")

        home_config = read_home_config()
        parsed_args = parser.parse_args(args)
        if not parsed_args.name:
            parser.print_help()
            sys.exit(1)

        path = os.getcwd()
        if parsed_args.directory:
            path = os.path.abspath(parsed_args.directory)

        if os.path.exists(path) and not os.path.isdir(path):
            sys.stderr.write('Path already exist and is not a directory: ' + path)

        if not os.path.exists(path):
            os.makedirs(path)

        yaml = ruamel.yaml.YAML()
        config = {}

        if os.path.exists(path+'/aetros.yml'):
            with open(path+'/aetros.yml', 'r') as f:
                config = yaml.load(f)

            if isinstance(config, dict) and 'model' in config and not parsed_args.force:
                print("failed: aetros.yml already exists in with a linked model to " + config['model']+ '. Use -f to force.')
                sys.exit(1)

        if not parsed_args.private:
            print("Warning: creating public model. Use --private to create private models.")

        if '/' in parsed_args.name:
            sys.stderr.write('No / allowed in name. Use -o if thie model should be created in an organisation.')
            sys.exit(1)

        response = api.create_model(parsed_args.name or (os.path.basename(os.getcwd())), parsed_args.organisation, parsed_args.space, parsed_args.private)
        name = response['name']

        if response['already_exists']:
            print("Notice: Model already exists remotely.")

        config['model'] = name

        with open(path + '/aetros.yml', 'w+') as f:
            yaml.dump(config, f)

        print("aetros.yml created and linked with model " + name + ' in ' + path)
        print("Open AETROS Trainer to see the model at https://" + home_config['host'] + '/model/' + name)

        git_remote_url = 'git@%s:%s.git' % (home_config['host'], name)

        print("Use git to store your source code. Each model has its own Git repository.")
        print("  $ cd " + path)
        print("  $ git init")
        print("  $ git remote add origin " + git_remote_url)
        print("  $ git add .")
        print("  $ git commit -m 'first commit'")
        print("  $ git push origin master")
