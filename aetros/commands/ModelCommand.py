from __future__ import absolute_import
from __future__ import print_function
import argparse

import sys
import os
from aetros.utils import find_config, find_config_path


class ModelCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' model')

        parsed_args = parser.parse_args(args)


        config_path = find_config_path()
        if not config_path:
            print("No model configuration file (aetros.yml). Switch to a directory first..")
            sys.exit(1)

        config = find_config(error_on_missing=True)
        print("Model %s in %s used in all aetros commands." % (config['model'], os.path.dirname(config_path)))
