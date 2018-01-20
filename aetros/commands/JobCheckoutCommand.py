from __future__ import absolute_import
from __future__ import print_function
import argparse
import subprocess

import sys

import os

from aetros.utils import read_home_config, setup_git_ssh, read_config, git_has_local_job, git_has_remote_job, \
    find_config


class JobCheckoutCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' job-checkout')
        parser.add_argument('job_id', help="Short or long job id like ed4d6a204")
        parser.add_argument('file', nargs='*', help="Checkout only one file.")
        parser.add_argument('--target', '-t', help="Target path where job files (or a single file) should be saved. Default current folder")
        parser.add_argument('--overwrite', '-p', help="Overwrite existing files.")
        parser.add_argument('--model', help="Model name like peter/mnist. Per default from current directory")
        parser.add_argument('-c', '--config', help="Default aetros.yml in current working directory.")

        parsed_args = parser.parse_args(args)

        if not parsed_args.job_id:
            parser.print_help()
            sys.exit()

        home_config = read_home_config()
        config = find_config(parsed_args.config)
        model = parsed_args.model if parsed_args.model else config['model']

        if not model:
            print("No model defined. Use --model or switch into a directory where you executed 'aetros init model-name'.")
            sys.exit(2)

        target = os.path.normpath(os.path.abspath(parsed_args.target if parsed_args.target else os.getcwd()))
        git_dir = os.path.normpath(home_config['storage_dir'] + '/' + model + '.git')

        if not parsed_args.file and not os.path.exists(target):
            os.makedirs(target)
        if parsed_args.file and not os.path.exists(target):
            os.makedirs(os.path.dirname(target))

        id_map = {}

        for job_id in [parsed_args.job_id]:
            full_id = git_has_local_job(home_config, model, job_id)
            id_map[job_id] = full_id
            if not full_id:
                full_id = git_has_remote_job(home_config, model, job_id)
                id_map[job_id] = full_id
                if full_id:
                    print("Pull job %s to local ... " % (job_id, ))
                    ref = 'refs/aetros/job/' + full_id
                    subprocess.call([home_config['git'], '--bare', '--git-dir', git_dir, 'fetch', 'origin', ref+':'+ref])
                else:
                    print("Job %s not found." % (job_id, ))
                    sys.exit(2)

        ref = 'refs/aetros/job/' + id_map[parsed_args.job_id]

        if not parsed_args.file:
            print("Checkout all job files %s %s into %s ... " % (model, id_map[parsed_args.job_id], target))
        else:
            print("Checkout job files %s %s into %s ... " % (model, id_map[parsed_args.job_id], target))

        paths = parsed_args.file if parsed_args.file else ['.']

        subprocess.call(
            [home_config['git'], '--bare', '--git-dir', git_dir, '--work-tree', target, 'checkout', ref, '--'] + paths
        )