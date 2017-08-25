import json
import subprocess
from six.moves.urllib.parse import urlencode
from aetros.utils import read_home_config

class ApiError(Exception):
    def __init__(self, message, reason):
        self.message = message
        self.reason = reason

    def __str__(self):
        return self.message + ', Reason: ' + self.reason


def request(path, query=None, body=None):
    query = query or {}

    if isinstance(query, dict):
        query = urlencode(query)

    config = read_home_config()

    if '?' in path:
        path += '&' + query
    else:
        path += '?' + query

    ssh_stream = subprocess.Popen(['ssh', 'git@' + config['host'], 'api', path],
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE if body is not None else None, stdout=subprocess.PIPE)

    if body is not None:
        ssh_stream.stdin.write(json.dumps(body))
        ssh_stream.stdin.close()

    stderr = read(ssh_stream.stderr)
    stdout = read(ssh_stream.stdout)

    ssh_stream.wait()

    if ssh_stream.returncode != 0:
        raise ApiError('Could not request api: ' + path, stderr)

    return stdout


def read(obj):
    r = ''
    while True:
        buf = obj.read()
        if buf == '':
            break
        r += buf

    return r


def parse_json(content):
    a = json.loads(content)

    if 'error' in a:
        raise Exception('API request failed %s: %s.' % (a['error'], a['message']))

    return a


def model(model_name):
    content = request('model', {'id': model_name}).decode('utf-8')

    return parse_json(content)


def create_job_info(model_name, parameters=None, dataset_id=None):
    content = request('job/create-info', {'id': model_name}, {'parameters': parameters, 'datasetId': dataset_id}).decode('utf-8')

    return parse_json(content)
