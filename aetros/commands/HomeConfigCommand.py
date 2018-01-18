from __future__ import absolute_import
from __future__ import print_function
import argparse

import os
import sys

import simplejson
from ruamel import yaml

class HomeConfigCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' home-config')
        parser.add_argument('name', nargs='?', help="Config name")
        parser.add_argument('value', nargs='?', help="Value")
        parsed_args = parser.parse_args(args)

        if not parsed_args.name:
            parser.print_help()
            sys.exit()

        path = os.getenv('AETROS_HOME_CONFIG_FILE') or '~/aetros.yml'
        path = os.path.normpath(os.path.expanduser(path))
        config = {}

        if os.path.exists(path):
            f = open(path, 'r')
            try:
                config = yaml.safe_load(f)
            except Exception:
                raise Exception('Could not load aetros home config at ' + os.path.realpath(path))

        if not config:
            config = {}

        json = ['ssl_verify', 'http_port', 'https_port', 'ssl', 'git_port']

        if parsed_args.name in json:
            parsed_args.value = simplejson.loads(parsed_args.value)

        config[parsed_args.name] = parsed_args.value

        with open(path, 'w+') as f:
            f.write(yaml.dump(config))

        print(yaml.dump(config)+'\n')

        print("Home config writte to " + path)
