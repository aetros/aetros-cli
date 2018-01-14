from __future__ import absolute_import
from __future__ import print_function
import argparse
from collections import OrderedDict

import six
from colorclass import Color
from terminaltables import AsciiTable

import sys

import os

from aetros.utils import read_home_config, setup_git_ssh, read_config, git_local_job_ids, git_remote_job_ids, \
    find_config


class JobsCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, prog=aetros.const.__prog__ + ' jobs')
        parser.add_argument('--local', '-l', action='store_true', help="Local only jobs view")
        parser.add_argument('--model', help="Model name like peter/mnist. Per default from configuration.")
        parser.add_argument('-c', '--config', help="Default aetros.yml in current working directory or directories above.")

        parsed_args = parser.parse_args(args)

        home_config = read_home_config()
        config = find_config(parsed_args.config)
        model = parsed_args.model if parsed_args.model else config['model']
        if not model:
            print("No model defined. Use --model or switch into a directory where a model is set up.")
            sys.exit(1)

        git_dir = os.path.normpath(home_config['storage_dir'] + '/' + model + '.git')
        git_remote_url = 'git@%s:%s.git' % (home_config['host'], model)

        if parsed_args.local and not os.path.isdir(git_dir):
            self.logger.error("Git repository for model %s in %s not found." % (parsed_args.id, git_dir))
            self.logger.error("You seem not to have any job created on this machine for model " + model)
            sys.exit(1)

        print("Show jobs of model " + model + ' ('+home_config['host']+')')

        setup_git_ssh(home_config)
        local_job_ids = git_local_job_ids(home_config, model)
        remote_job_ids = []

        try:
            remote_job_ids = git_remote_job_ids(home_config, model)
        except:
            pass

        job_map = OrderedDict()
        for job_id in local_job_ids:
            job_map[job_id] = {'local': Color('{autogreen}Yes{/autogreen}'), 'remote': Color('{autored}No{/autored}'),}

        for job_id in remote_job_ids:
            if job_id in job_map:
                job_map[job_id]['remote'] = Color('{autogreen}Yes{/autogreen}')
            elif parsed_args.local:
                job_map[job_id] = {'local': Color('{autored}No{/autored}'), 'remote': Color('{autogreen}Yes{/autogreen}')}

        print("%d jobs found. (%d synced to remote)" % (len(job_map), len(remote_job_ids)))

        table_data = [['Short Job ID', 'Local', 'Remote', 'Long Job ID']]

        for job_id, info in six.iteritems(job_map):
            table_data.append([job_id[0:9], info['local'], info['remote'], job_id])

        table = AsciiTable(table_data)
        print(table.table)