import subprocess

import re


def get_branches():
    branches = subprocess.check_output(['git', 'branch']).decode("utf-8").strip().split('\n')

    return [x.strip(' *') for x in branches]


def get_current_branch():
    branches = subprocess.check_output(['git', 'branch']).decode("utf-8")
    m = re.search('\* ([^\s]+)', branches)
    current_branch = m.group(1) if m else None

    return current_branch


def get_current_commit_hash():
    commit_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8")

    return commit_sha.strip()