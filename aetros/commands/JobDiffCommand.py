from __future__ import absolute_import
from __future__ import print_function
import argparse
import subprocess

import sys

import os

from aetros.utils import read_home_config, setup_git_ssh, read_config, git_has_local_job, git_has_remote_job, \
    find_config


class JobDiffCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' job-diff')
        parser.add_argument('id_from', help="Short or long job id like ed4d6a204.")
        parser.add_argument('id_to', nargs='?', help="Short or long job id like d55df24a7 or file path")
        parser.add_argument('limit', nargs='?', help="Limit files to diff")
        parser.add_argument('--model', help="Model name like peter/mnist. Per default from configuration.")
        parser.add_argument('-c', '--config', help="Default aetros.yml in current working directory or directories above.")

        parsed_args = parser.parse_args(args)

        home_config = read_home_config()
        config = find_config(parsed_args.config)
        model = parsed_args.model if parsed_args.model else config['model']

        if not model:
            print("No model defined. Use --model or switch into a directory where you executed 'aetros init model-name'.")
            sys.exit(2)

        git_dir = os.path.normpath(home_config['storage_dir'] + '/' + model + '.git')

        id_map = {}

        for job_id in [parsed_args.id_from, parsed_args.id_to]:
            if os.path.exists(job_id):
                continue

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

        print("Diff jobs %s and %s of %s." %(parsed_args.id_from, parsed_args.id_to, model))

        from_ref = 'refs/aetros/job/' + id_map[parsed_args.id_from]
        args = [home_config['git'], '--bare', '--git-dir', git_dir]

        if os.path.exists(parsed_args.id_to):
            args += ['--work-tree', os.path.abspath(parsed_args.id_to), 'diff', from_ref]
        else:
            to_ref = 'refs/aetros/job/' + id_map[parsed_args.id_to]
            args += ['diff', from_ref+'...'+to_ref]

        if parsed_args.limit:
            args += ['--', parsed_args.limit]

        subprocess.call(args)