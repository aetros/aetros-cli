from __future__ import absolute_import
from __future__ import print_function
import argparse
import subprocess

import sys

import os

from aetros.utils import read_home_config, setup_git_ssh, read_config, git_has_local_job, git_has_remote_job, \
    find_config


class JobFilesCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' job-files')
        parser.add_argument('job_id', help="Short or long job id like ed4d6a204")
        parser.add_argument('folder', nargs='?', help="Limit files list to folder. Default root ./")
        parser.add_argument('-r', action='store_true', help="Recursive files tree")
        parser.add_argument('--model', help="Model name like peter/mnist. Per default from found configuration.")
        parser.add_argument('-c', '--config', help="Default aetros.yml in current working directory or directories above.")

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

        git_dir = os.path.normpath(home_config['storage_dir'] + '/' + model + '.git')

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

        print("List job files of %s of %s" % (parsed_args.job_id, model))
        args = [home_config['git'], '--bare', '--git-dir', git_dir, 'ls-tree', '--long']
        if parsed_args.r:
            args.append('-r')

        args.append(ref)

        if parsed_args.folder:
            args.append(parsed_args.folder)
        subprocess.call(args)