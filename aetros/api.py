import subprocess

import six
from ruamel import yaml
from six.moves.urllib.parse import urlencode
import json

from aetros.logger import drain_stream
from aetros.utils import read_home_config, create_ssh_stream


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

    if method == 'get' and body is not None:
        method = 'post'

    ssh_stream = create_ssh_stream(config)
    stdin, stdout, stderr = ssh_stream.exec_command('api ' + method + ' ' + path)

    if body is not None:
        input = six.b(json.dumps(body))
        stdin.write(input)

    stdout = drain_stream(stdout)
    stderr = drain_stream(stderr)

    if len(stderr) > 0:
        if six.b('Connection ') in stderr:
            raise ApiConnectionError('Could not request api: ' + path, stderr)

        if six.b('Permission denied') in stderr:
            raise ApiConnectionError('Could not request API due to permission denied. '
                                     'Did you setup authentication properly?', stderr)

        raise ApiError('Could not request api: ' + path, stderr)

    return stdout


def raise_response_exception(message, response):
    content = response.content
    error = 'unknown'
    error_message = ''

    if isinstance(content, six.string_types):
        try:
            content_parsed = yaml.safe_load(content)
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
        a = yaml.safe_load(content)
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


def create_job(model_name, local=False, parameters=None, dataset_id=None, config=None):
    content = request(
        'job',
        {'modelId': model_name},
        {'parameters': parameters, 'datasetId': dataset_id, 'config': config, 'local': local},
        'put'
    )
    return parse_json(content)


def create_model(model_name, private = False):
    content = request('model/create', {'name': model_name, 'private': private}, None, 'put')

    return parse_json(content)
