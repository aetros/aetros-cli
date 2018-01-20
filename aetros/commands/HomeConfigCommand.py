from __future__ import absolute_import
from __future__ import print_function
import argparse

import os
import sys

import simplejson
import ruamel.yaml

class HomeConfigCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        description = 'aetros home-config --reset\naetros home-config --delete host\naetros home-config host localhost'

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' home-config', description=description)
        parser.add_argument('name', nargs='?', help="Config name")
        parser.add_argument('--reset', '-r', action='store_true', help="Resets the configuration file, making defaults active.")
        parser.add_argument('--delete', '-d', action='store_true', help="Delete the value of name")
        parser.add_argument('value', nargs='?', help="Value")
        parsed_args = parser.parse_args(args)

        path = os.getenv('AETROS_HOME_CONFIG_FILE') or '~/aetros.yml'
        path = os.path.normpath(os.path.expanduser(path))
        config = {}

        yaml = ruamel.yaml.YAML()

        if parsed_args.reset:
            with open(path, 'w+') as f:
                f.write("")

            print("File %s has been reset." % (path,))
            sys.exit(0)

        if not parsed_args.name:
            parser.print_help()
            sys.exit()

        if os.path.exists(path):
            f = open(path, 'r')
            try:
                config = yaml.load(f)
            except Exception:
                print('Error: could not load aetros home config at ' + os.path.realpath(path))
                raise

        if not config:
            config = {}

        json = ['ssl_verify', 'http_port', 'https_port', 'ssl', 'ssh_port']

        if parsed_args.delete:
            if parsed_args.value:
                print("Error: no value allowed when using --delete")
                sys.exit(1)

            if parsed_args.name in config:
                del config[parsed_args.name]

        else:
            if parsed_args.name in json:
                parsed_args.value = simplejson.loads(parsed_args.value)

            config[parsed_args.name] = parsed_args.value

        with open(path, 'w+') as f:
            yaml.dump(config, f)

        yaml.dump(config, sys.stdout)
        print("\nHome config written to " + path)
