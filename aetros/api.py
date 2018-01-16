import collections
import requests
import six
import sys

from requests.auth import HTTPBasicAuth
from six.moves.urllib.parse import urlencode
import simplejson

from aetros.logger import drain_stream
from aetros.utils import read_home_config, create_ssh_stream

class ApiError(Exception):
    def __init__(self, message, error):
        self.message = message
        self.error = error

    def __str__(self):
        return self.message + ', Reason: ' + self.error


class ApiConnectionError(ApiError):
    pass


def http_request(path, query='', json_body=None, method='get', config=None):
    config = read_home_config() if config is None else config

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception: pass

    if query is not None:
        if isinstance(query, dict):
            query = urlencode(query)

        if '?' in path:
            path += '&' + query
        else:
            path += '?' + query

    url = 'https://' + config['host'] + '/api/' + path
    auth = None
    if 'auth_user' in config:
        auth = HTTPBasicAuth(config['auth_user'], config['auth_pw'])

    if json_body is not None and method == 'get':
        method = 'post'

    response = requests.request(
        method, url, data=json_body,
        auth=auth, verify=config['ssl_verify'],
        headers={'Accept': 'application/json'}
    )

    if response.status_code >= 400:
        raise_response_exception('Failed request ' + path, response)

    return parse_json(response.content.decode('utf-8'))


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
    stdin, stdout, stderr = ssh_stream.exec_command('api ' + method + ' ' + simplejson.dumps(path))

    if body is not None:
        input = six.b(simplejson.dumps(body))
        stdin.write(input)
        stdin.flush()
        stdin.channel.shutdown_write()

    stdout = drain_stream(stdout)
    stderr = drain_stream(stderr)

    if len(stderr) > 0:
        stderr = stderr.decode('utf-8')

        raise ApiError('Could not request api: ' + path, stderr)

    return stdout


def raise_response_exception(message, response):
    content = response.content
    error = 'unknown'
    error_message = ''

    if isinstance(content, six.string_types):
        try:
            content_parsed = simplejson.loads(content, object_pairs_hook=collections.OrderedDict)
            if 'error' in content_parsed:
                error = content_parsed['error']
            if 'message' in content_parsed:
                error_message = content_parsed['message']
        except Exception: pass

    reason = 'StatusCode='+str(response.status_code)+', error: ' + str(error)+ ', message: ' + str(error_message)

    raise ApiError(message, reason)


def parse_json(content):
    a = simplejson.loads(content)

    if isinstance(a, dict) and 'error' in a:
        raise ApiError('API request failed %s: %s.' % (a['error'], a['message']), a['error'])

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
    content = request('model/create', None, {'name': model_name, 'private': private}, 'put')
    return parse_json(content)
