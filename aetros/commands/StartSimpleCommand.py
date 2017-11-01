from __future__ import absolute_import
from __future__ import print_function
import argparse

import sys

from aetros.starter import start_keras
from aetros.backend import JobBackend
from aetros.utils import unpack_full_job_id


class StartSimpleCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' run', description="Internal usage.")

        parser.add_argument('id', nargs='?', help='Job id, e.g. user/modelname/0db75a64acb74c27bd72c22e359de7a4c44a20e5 to start a pre-created job.')
        parser.add_argument('--fetch', action='store_true', help="Fetch job from server.")

        parsed_args = parser.parse_args(args)

        if not parsed_args.id:
            parser.print_help()
            sys.exit(1)

        owner, name, id = unpack_full_job_id(parsed_args.id)

        job_backend = JobBackend(model_name=owner + '/' + name)
        job_backend.section('checkout')

        if parsed_args.fetch:
            job_backend.fetch(id)

        job_backend.load(id)
        job_backend.start()

        start_keras(self.logger, job_backend)
