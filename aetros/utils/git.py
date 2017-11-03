import subprocess

import re

import os


def silent_execute(args):
    try:
        with open(os.devnull, 'r+b', 0) as DEVNULL:
            return subprocess.check_output(args, stderr=DEVNULL).decode("utf-8").strip()
    except Exception:
        return None


def get_branches():
    output = silent_execute(['git', 'branch'])
    if output:
        branches = output.split('\n')
    
        return [x.strip(' *') for x in branches]


def get_current_branch():
    output = silent_execute(['git', 'branch'])
    if output:
        m = re.search('\* ([^\s]+)', output)
        current_branch = m.group(1) if m else None

        return current_branch


def get_current_remote_url(origin_name = 'origin'):
    output = silent_execute(['git', 'remote', '-v'])
    if output:
        import re
        regex = re.compile('^' + re.escape(origin_name) + '\t([^\s]+)', re.MULTILINE)
        matches = [m.groups() for m in regex.finditer(output)]
        if matches:
            return matches[0][0]

        return None


def get_current_commit_hash():
    return silent_execute(['git', 'rev-parse', 'HEAD'])


def get_current_commit_message():
    return silent_execute(['git', 'log', '-1', '--pretty=%B', 'HEAD'])


def get_current_commit_author():
    return silent_execute(['git', 'log', '-1', '--pretty=%an <%ae>', 'HEAD'])
