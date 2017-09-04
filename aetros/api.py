import subprocess

import six
from six.moves.urllib.parse import urlencode
from aetros.utils import read_config
import json

class ApiError(Exception):
    def __init__(self, message, reason):
        self.message = message
        self.reason = reason

    def __str__(self):
        return self.message + ', Reason: ' + self.reason

class ApiConnectionError(ApiError):
    pass

def request(path, query=None, body=None, config=None):
    query = query or {}

    if isinstance(query, dict):
        query = urlencode(query)

    if '?' in path:
        path += '&' + query
    else:
        path += '?' + query

    config = read_config() if config is None else config

    args = [config['ssh']] if isinstance(config['ssh'], six.string_types) else config['ssh']
    args += ['-o', 'StrictHostKeyChecking no']

    if config['ssh_key']:
        args += ['-i', config['ssh_key']]

    ssh_stream = subprocess.Popen(args + ['git@' + config['host'], 'api', path],
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE if body is not None else None, stdout=subprocess.PIPE)

    if body is not None:
        ssh_stream.stdin.write(json.dumps(body))
        ssh_stream.stdin.close()

    stderr = read(ssh_stream.stderr)
    stdout = read(ssh_stream.stdout)

    ssh_stream.wait()

    if ssh_stream.returncode != 0:
        if 'Connection ' in stderr:
            raise ApiConnectionError('Could not request api: ' + path, stderr)

        raise ApiError('Could not request api: ' + path, stderr)

    return stdout.decode('utf8')


def raise_response_exception(message, response):

    content = response.content
    error = 'unknown'
    error_message = ''

    if isinstance(content, six.string_types):
        try:
            content_parsed = json.loads(content)
            if 'error' in content_parsed:
                error = content_parsed['error']
            if 'message' in content_parsed:
                error_message = content_parsed['message']
        except:
            pass

    reason = 'StatusCode='+str(response.status_code)+', error: ' + str(error)+ ', message: ' + str(error_message)

    raise ApiError(message, reason)


def read(obj):
    r = six.b('')
    while True:
        buf = obj.read()
        if buf == six.b(''):
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
