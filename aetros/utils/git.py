import subprocess

import re

import os


def get_branches():
    with open(os.devnull, 'r+b', 0) as DEVNULL:
        branches = subprocess.check_output(['git', 'branch'], stderr=DEVNULL).decode("utf-8").strip().split('\n')

        return [x.strip(' *') for x in branches]


def get_current_branch():
    with open(os.devnull, 'r+b', 0) as DEVNULL:
        branches = subprocess.check_output(['git', 'branch'], stderr=DEVNULL).decode("utf-8")
        m = re.search('\* ([^\s]+)', branches)
        current_branch = m.group(1) if m else None

        return current_branch


def get_current_commit_hash():
    with open(os.devnull, 'r+b', 0) as DEVNULL:
        commit_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=DEVNULL).decode("utf-8")

        return commit_sha.strip()