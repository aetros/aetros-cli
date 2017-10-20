from __future__ import absolute_import
from __future__ import print_function
import argparse

import sys

import six

from aetros.backend import JobBackend
from aetros import api
from aetros.utils import read_config, human_size

class IdCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' run')

        parsed_args = parser.parse_args(args)

        user = api.user()

        print("Key installed of account %s (%s)" % (user['username'], user['name']))

        if len(user['accounts']) > 0:
            for orga in six.itervalues(user['accounts']):
                print("  %s of organisation %s (%s)." % ("Owner" if orga['memberType'] == 1 else "Member", orga['username'], orga['name']))
        else:
            print("  Without membership to an organisation.")
