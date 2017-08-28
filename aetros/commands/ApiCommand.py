from __future__ import absolute_import
from __future__ import print_function
import argparse

import sys

from aetros import api

class ApiCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                description="You can provide json in stdin to issue a POST call", prog=aetros.const.__prog__ + ' run')
        parser.add_argument('path', nargs='?', help="Request path + query, e.g. model/settings?name=owner/name")
        parser.add_argument('--method', nargs='?', help="Per default GET, if stdin data is given a POST. Alternatively provide a HTTP verb.")

        parsed_args = parser.parse_args(args)

        if not parsed_args.path:
            parser.print_help()
            sys.exit(1)

        sys.stdout.write(api.request(parsed_args.path))