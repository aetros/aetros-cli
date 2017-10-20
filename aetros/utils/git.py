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


def get_current_remote_url(origin_name = 'origin'):
    with open(os.devnull, 'r+b', 0) as DEVNULL:
        output = subprocess.check_output(['git', 'remote', '-v'], stderr=DEVNULL).decode("utf-8").strip()

        import re
        match = re.match('^' + re.escape(origin_name) + '\t([^\s]+)', output)
        if match:
            return match.group(1)

        return output.strip()


def get_current_commit_hash():
    with open(os.devnull, 'r+b', 0) as DEVNULL:
        output = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=DEVNULL).decode("utf-8")

        return output.strip()


def get_current_commit_message():
    with open(os.devnull, 'r+b', 0) as DEVNULL:
        output = subprocess.check_output(['git', 'log', '-1', '--pretty=%B', 'HEAD'], stderr=DEVNULL).decode("utf-8")

        return output.strip()


def get_current_commit_author():
    with open(os.devnull, 'r+b', 0) as DEVNULL:
        output = subprocess.check_output(['git', 'log', '-1', '--pretty=%an <%ae>', 'HEAD'], stderr=DEVNULL).decode("utf-8")

        return output.strip()