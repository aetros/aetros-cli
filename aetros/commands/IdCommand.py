from __future__ import absolute_import
from __future__ import print_function
import argparse

import six
import sys

from aetros import api
from aetros.utils import read_home_config, KeyNotConfiguredException


class IdCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' id')

        parsed_args = parser.parse_args(args)
        config = read_home_config()


        try:
            user = api.user()
        except KeyNotConfiguredException as e:
            self.logger.error(str(e))
            sys.exit(1)

        print("Logged in as %s (%s) on %s" % (user['username'], user['name'], config['host']))

        if len(user['accounts']) > 0:
            for orga in six.itervalues(user['accounts']):
                print("  %s of organisation %s (%s)." % ("Owner" if orga['memberType'] == 1 else "Member", orga['username'], orga['name']))
        else:
            print("  Without membership to an organisation.")
