import subprocess

import six
from six.moves.urllib.parse import urlencode
import json

from aetros.utils import read_home_config


class ApiError(Exception):
    def __init__(self, message, reason):
        self.message = message
        self.reason = reason

    def __str__(self):
        return self.message + ', Reason: ' + self.reason


class ApiConnectionError(ApiError):
    pass


def request(path, query=None, body=None, method='get', config=None):
    query = query or {}

    if isinstance(query, dict):
        query = urlencode(query)

    if '?' in path:
        path += '&' + query
    else:
        path += '?' + query

    config = read_home_config() if config is None else config

    args = [config['ssh']] if isinstance(config['ssh'], six.string_types) else config['ssh']
    args += ['-o', 'StrictHostKeyChecking no']

    if config['ssh_key']:
        args += ['-i', config['ssh_key']]

    if method == 'get' and body is not None:
        method = 'post'

    ssh_stream = subprocess.Popen(args + ['git@' + config['host'], 'api', method, path],
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE if body is not None else None, stdout=subprocess.PIPE)

    input = None
    if body is not None:
        input = six.b(json.dumps(body))

    stdout, stderr = ssh_stream.communicate(input)

    if ssh_stream.returncode != 0:
        if 'Connection ' in stderr:
            raise ApiConnectionError('Could not request api: ' + path, stderr)

        raise ApiError('Could not request api: ' + path, stderr)

    return stdout.decode('utf-8')


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
    r = b''
    while True:
        buf = obj.read()
        if buf == b'':
            break
        r += buf

    return r.decode("utf-8")


def parse_json(content):
    try:
        a = json.loads(content)
    except:
        raise Exception('Could not parse api request. Content: ' + str(content))

    if 'error' in a:
        raise Exception('API request failed %s: %s.' % (a['error'], a['message']))

    return a


def model(model_name):
    content = request('model', {'id': model_name}).decode('utf-8')

    return parse_json(content)


def user():
    return parse_json(request('user'))


def create_job(model_name, server=None, parameters=None, dataset_id=None, config=None):
    content = request(
        'job',
        {'modelId': model_name},
        {'server': server, 'parameters': parameters, 'datasetId': dataset_id, 'config': config},
        'put'
    )
    return parse_json(content)


def create_model(model_name, private = False):
    content = request('model/create', {'name': model_name, 'private': private}, None, 'put')

    return parse_json(content)
