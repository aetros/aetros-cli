from __future__ import division
from __future__ import absolute_import

import os
import re
import time
import datetime

import numpy as np
import signal

import paramiko
import six
import sys
import ruamel.yaml as yaml

start_time = time.time()
last_time = None

def stop_time(title=''):
    global last_time
    diff = ("{0:.10f}".format(time.time() - last_time)) if last_time is not None else ''
    last_time = time.time()
    sys.__stdout__.write("STOP_TIME: " + str(time.time()-start_time) + "s - diff: "+diff+"  - " +title+ "\n")


def get_option(dict, key, default=None, type=None):
    if key not in dict or dict[key] == '':
        return default

    if type == 'bool':
        bool(dict[key])

    return dict[key]


def unpack_simple_job_id(full_id):
    [owner, model, id] = unpack_full_job_id(full_id)

    return [owner + '/' + model, id]


def unpack_full_job_id(full_id):
    if full_id.count('/') < 2:
        raise Exception('Not a valid full job id: ' + full_id)

    owner = full_id[0:full_id.index('/')]
    model = full_id[full_id.index('/') + 1:full_id.rindex('/')]
    id = full_id[full_id.rindex('/')+1:]

    return [owner, model, id]


def create_ssh_stream(config, exit_on_failure=True):
    ssh_stream = paramiko.client.SSHClient()
    # ssh_stream.load_system_host_keys()
    ssh_stream.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    key = None
    key_filename = get_ssh_key_for_host(config['host'])

    if config['ssh_key_base64']:
        key_filename = None
        key = paramiko.RSAKey.from_private_key(six.StringIO(config['ssh_key_base64']))

    if not key and not key_filename:
        raise Exception("No SSH key configured for " + config['host'] + ". See https://aetros.com/docu/trainer/authentication")

    key_description = key_filename if key_filename else 'from server'

    try:
        ssh_stream.connect(config['host'], key_filename=key_filename, username='git', compress=True, pkey=key)
    except Exception as e:
        if isinstance(e, paramiko.ssh_exception.AuthenticationException) or isinstance(e, paramiko.ssh_exception.SSHException):
            if exit_on_failure:
                print("Fatal: AETROS authentication against "+config['host']+" failed using key "+key_description+": "+str(e)+
                      ". Did you setup SSH keys correctly? See https://aetros.com/docu/trainer/authentication")
                sys.exit(1)
        raise

    return ssh_stream


def setup_git_ssh(config):
    import tempfile
    ssh_command = config['ssh']
    ssh_command += ' -o StrictHostKeyChecking=no'

    ssh_key = None
    if config['ssh_key_base64']:
        ssh_key = tempfile.NamedTemporaryFile(delete=False, prefix='ssh_key_')
        ssh_key.write(six.b(config['ssh_key_base64']))
        ssh_key.close()
        ssh_command += ' -i '+ ssh_key.name
        os.chmod(ssh_key.name, 0o600)
    # elif config['ssh_key']:
    #     ssh_command += ' -i '+ os.path.expanduser(config['ssh_key'])

    ssh_script = tempfile.NamedTemporaryFile(delete=False, prefix='git_ssh_')
    ssh_script.write(six.b(ssh_command + ' "$@"'))
    ssh_script.close()
    os.environ['GIT_SSH'] = ssh_script.name
    os.chmod(ssh_script.name, 0o700)

    def delete():
        if os.path.exists(ssh_script.name):
            os.unlink(ssh_script.name)

        if ssh_key and os.path.exists(ssh_key.name):
            os.unlink(ssh_key.name)

    return delete


def get_ssh_key_for_host(host):
    ssh_config = paramiko.SSHConfig()
    user_config_file = os.path.expanduser("~/.ssh/config")
    if os.path.exists(user_config_file):
        with open(user_config_file) as f:
            ssh_config.parse(f)

    user_config = ssh_config.lookup(host)

    if 'identityfile' in user_config:
        path = os.path.expanduser(user_config['identityfile'][0])
        if not os.path.exists(path):
            raise Exception("Specified IdentityFile "+path
                            + " for " + host + " in ~/.ssh/config not existing anymore.")

        return path


def read_home_config(path = None, logger=None):
    if not path:
        path = os.getenv('AETROS_HOME_CONFIG_FILE') or '~/aetros.yml'

    path = os.path.normpath(os.path.expanduser(path))
    custom_config = {}

    if os.path.exists(path):
        f = open(path, 'r')
        try:
            logger and logger.debug('Home config loaded from ' + os.path.realpath(path))
            custom_config = yaml.safe_load(f)
        except Exception:
            raise Exception('Could not load aetros home config at ' + os.path.realpath(path))

        if custom_config is None:
            custom_config = {}

    config = {
        'host': os.getenv('AETROS_HOST') or 'trainer.aetros.com',
        'storage_dir': os.getenv('AETROS_STORAGE_DIR') or '~/.aetros',
        'ssh_key_base64': os.getenv('AETROS_SSH_KEY_BASE64'),
        'image': None,
        'ssh': 'ssh',
        'git': 'git',
        'docker': 'docker',
        'docker_options': [],
        'ssl_verify': True,
    }

    config.update(custom_config)

    config['storage_dir'] = os.path.abspath(os.path.expanduser(config['storage_dir']))

    return config

ignore_pattern_cache = {}


def extract_parameters(full_definitions, overwritten = None, incoming_path = ''):
    container = {}

    if overwritten is None:
        overwritten = {}

    for parameter in full_definitions:
        param_type = parameter['type']

        if 'defaultValue' in parameter:
            defaultValue = parameter['defaultValue']
        elif 'default' in parameter:
            defaultValue = parameter['default']
        else:
            defaultValue = None

        value = defaultValue
        if 'value' in parameter:
            value = parameter['value']

        name = parameter['name']

        path = (incoming_path + '.' + name).strip('.')

        if path in overwritten:
            value = overwritten[path]

        if param_type == 'string':
            container[name] = str(value)
#
        if param_type == 'number':
            container[name] = float(value)
#
        if param_type == 'boolean':
            container[name] = bool(value)
#
        if param_type == 'array':
            container[name] = value

        if param_type == 'group':
            container[name] = extract_parameters(parameter['children'], overwritten, path)

        if param_type == 'choice_group':
            if '' == value:
                #no default value nor value set
                continue

            if isinstance(value, six.string_types) and value not in parameter['children']:
                found_names = []
                for idx, child in enumerate(parameter['children']):
                    found_names.append(child['name'])
                    if value == child['name']:
                        value = idx
                        break

                if isinstance(value, six.string_types):
                    names = ', '.join(found_names)
                    raise Exception(str(value) + " is not available in " + path + ". " + names)

            value_name = parameter['children'][value]['name']
            container[name] = {
                '$value': value_name,
            }
            container[name][value_name] = extract_parameters(parameter['children'][value]['children'], overwritten, path)

        if param_type == 'choice_string' or param_type == 'choice_number':
            if path in overwritten:
                for idx, child in enumerate(parameter['children']):
                    if overwritten[path] == child['value']:
                        value = idx
                        break

            if value is None or (value < 0 or value >= len(parameter['children'])):
                value = 0

            container[name] = parameter['children'][value]['value']

    return container


def lose_parameters_to_full(parameters):
    if not parameters:
        return {}

    if isinstance(parameters, list):
        # full definition with type, name etc.
        return parameters

    def type_of(v):
        if isinstance(v, six.string_types):
            return 'string'

        if isinstance(v, (float, int)):
            return 'number'

        if isinstance(v, bool):
            return 'boolean'

        if isinstance(v, (list, dict)):
            return 'array'

    def extract_from_parameters(parameters):
        definitions = []

        for k, v in six.iteritems(parameters):
            definition = {'name': k}

            if k[0] != '_' and isinstance(v, list):
                # //regular array ['a', 'b']
                if not v:
                    definition['type'] = 'choice_string'
                    continue
                if isinstance(v[0], six.string_types):
                    definition['type'] = 'choice_string'
                else:
                    definition['type'] = 'choice_number'

                definition['children'] = []
                definition['defaultValue'] = 0

                for child in v:
                    definition['children'].append({'value': child})

            elif k[0] != '_' and isinstance(v, dict):
                # named keys, so a group
                # ['a' => 'bla']
                # becomes a choicegroup when all childern are groups as well
                all_groups = len(v) > 1

                for children in six.itervalues(v):
                    if not isinstance(children, dict):
                        all_groups = False
                        break

                if all_groups:
                    # when group in group, then first become choice_group
                    definition['type'] = 'choice_group'
                    definition['defaultValue'] = 0
                else:
                    definition['type'] = 'group'

                definition['children'] = extract_from_parameters(v)
            else:
                definition['type'] = type_of(v)
                definition['defaultValue'] = v

            definitions.append(definition)

        return definitions

    return extract_from_parameters(parameters)


def read_parameter_by_path(dictionary, path, return_group=False):

    # if path in dictionary:
    #     return dictionary[path]

    # elif '.' not in path:
    #     raise Exception('Parameter ' + path + ' not found and no default value given.')

    if not dictionary:
        return None

    current_group = None

    current = dictionary

    for item in path.split('.'):
        current_group = None
        if item not in current:
            raise Exception('Parameter ' + str(path) + ' not found and no default value given.')

        current = current[item]

        if isinstance(current, dict) and '$value' in current and current['$value'] in current:
            current_group = current
            current = current[current['$value']]

    if isinstance(current_group, dict) and '$value' in current_group and current_group['$value'] in current_group:
        if return_group:
            return current_group[current_group['$value']]
        else:
            return current_group['$value']

    return current


def flatten_parameters(parameters, incomingPath = ''):
    result = {}

    for key, value in six.iteritems(parameters):
        if '$value' == key: continue

        path = (incomingPath + '.' + key).strip('.')

        if isinstance(value, dict) and key[0] != '_':
            if '$value' in value:
                result[path] = value['$value']
                if value['$value'] not in value:
                    raise Exception(value['$value'] + " of " + path + " not in " + str(value))
                value = value[value['$value']]
                result.update(flatten_parameters(value, path))
            else:
                result.update(flatten_parameters(value, path))

        elif isinstance(value, list) and key[0] != '_':
            result[path] = value[0]
        else:
            result[path] = value

    return result


def is_ignored(path, ignore_patters):
    if isinstance(ignore_patters, six.string_types):
        ignore_patters = ignore_patters.split('\n')

    ignored = None

    if not ignore_patters:
        return False

    for pattern in ignore_patters:
        if not pattern:
            continue

        if pattern in ignore_pattern_cache:
            reobj = ignore_pattern_cache[pattern]
        else:
            if pattern[0] == '!':
                regex = re.escape(pattern[1:])
            else:
                regex = re.escape(pattern)

            regex = regex.replace('\\*\\*', '([^/\\\\]+[//\\\\])+([^/\\\\]+)')
            regex = regex.replace('\\*', '[^/\\\\]+')
            regex = '(' + regex + ')'
            if pattern[0] == '/' or (pattern[0] == '!' and pattern[1] == '/'):
                regex = '^' + regex
            else:
                regex = '^.*' + regex

            reobj = re.compile(regex)
            ignore_pattern_cache[pattern] = reobj

        normalized_path = path

        if pattern.startswith('!'):
            if pattern[1] == '/' and path[0] != '/':
                normalized_path = '/' + path

            # whitelist begins with !
            if reobj.match(normalized_path):
                ignored = False
        else:
            if pattern[0] == '/' and path[0] != '/':
                normalized_path = '/' + path

            if reobj.match(normalized_path):
                ignored = True

    if ignored is None:
        return False

    return ignored


def read_config(path = 'aetros.yml', logger=None):
    path = os.path.normpath(os.path.expanduser(path))

    config = {
        'dockerfile': None,
        'command': None,
        'install': None,
        'ignore': None,
        'image': None,
        'server': None,
        'parameters': {},
        'servers': None,
        'before_command': [],
    }

    if os.path.exists(path):
        f = open(path, 'r')

        custom_config = yaml.safe_load(f)
        if custom_config is None:
            custom_config = {}

        if 'storage_dir' in custom_config:
            del custom_config['storage_dir']

        config.update(custom_config)

        logger and logger.debug('Config loaded from ' + os.path.realpath(path))

    if 'parameters' not in config:
        config['parameters'] = {}

    return config


def raise_sigint():
    """
    Raising the SIGINT signal in the current process and all sub-processes.

    os.kill() only issues a signal in the current process (without subprocesses).
    CTRL+C on the console sends the signal to the process group (which we need).
    """
    if hasattr(signal, 'CTRL_C_EVENT'):
        # windows. Need CTRL_C_EVENT to raise the signal in the whole process group
        os.kill(os.getpid(), signal.CTRL_C_EVENT)
    else:
        # unix.
        pgid = os.getpgid(os.getpid())
        if pgid == 1:
            os.kill(os.getpid(), signal.SIGINT)
        else:
            os.killpg(os.getpgid(os.getpid()), signal.SIGINT)


def prepend_signal_handler(sig, f):

    previous_handler = None
    if callable(signal.getsignal(sig)):
        previous_handler = signal.getsignal(sig)

    def execute_signal_handler(*args, **kwargs):
        f(*args, **kwargs)

        # previous handles comes after this, so we actually prepend
        if previous_handler:
            previous_handler(*args, **kwargs)

    signal.signal(sig, execute_signal_handler)


def invalid_json_values(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bytes):
        return obj.decode('cp437')

    if isinstance(map, type) and isinstance(obj, map):
        # python 3 map
        return list(obj)

    raise TypeError('Invalid data type passed to json encoder: ' + type(obj).__name__)


def get_time(self):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    return st


def get_time_with_milli(self):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    return st + '.' + str(ts % 1)[2:6]


def human_size(size_bytes, precision=0):
    """
    Format a size in bytes into a 'human' file size, e.g. bytes, KB, MB, GB, TB, PB
    Note that bytes/KB will be reported in whole numbers but MB and above will have greater precision
    e.g. 1 byte, 43 bytes, 443 KB, 4.3 MB, 4.43 GB, etc
    """
    if size_bytes == 1:
        # because I really hate unnecessary plurals
        return "1 byte"

    suffixes_table = [('bytes',0),('KB',0),('MB',1),('GB',2),('TB',2), ('PB',2)]

    num = float(size_bytes)
    for suffix, precision in suffixes_table:
        if num < 1024.0:
            break
        num /= 1024.0

    if precision == 0:
        formatted_size = "%d" % num
    else:
        formatted_size = str(round(num, ndigits=precision))

    return "%s %s" % (formatted_size, suffix)


def array_to_img(x, scale=True):
    """
    x should be shape (channels, width, height)
    """
    from PIL import Image
    if x.ndim != 3:
        raise Exception('Unsupported shape : ', str(x.shape), '. Need (channels, width, height)')
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[0] == 3:
        # RGB
        if x.dtype != 'uint8':
            x = x.astype('uint8')
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[0] == 1:
        # grayscale
        if x.dtype != 'uint8':
            x = x.astype('uint8')
        return Image.fromarray(x.reshape(x.shape[1], x.shape[2]), 'L')
    else:
        raise Exception('Unsupported channel number: ', x.shape[0])
