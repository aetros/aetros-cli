from __future__ import division
from __future__ import absolute_import

import logging
import os
import re
import time
import datetime
import traceback
from threading import Thread

import numpy as np
import signal

import paramiko
import six
import sys
import ruamel.yaml as yaml
import subprocess

from paramiko.compress import ZlibCompressor

start_time = time.time()
last_time = None


def stop_time(title=''):
    global last_time
    diff = ("{0:.10f}".format(time.time() - last_time)) if last_time is not None else ''
    last_time = time.time()
    sys.__stdout__.write("STOP_TIME: " + str(time.time()-start_time) + "s - diff: "+diff+"  - " +title+ "\n")


def is_debug2():
    return os.getenv('DEBUG') == '2'


def is_debug():
    return is_debug2() or os.getenv('DEBUG') == '1'


def get_logger(name='', debug=None, format=None):

    import coloredlogs
    # logging.basicConfig() # this will make paramiko log a lot of stuff
    logger = logging.getLogger(name if name else 'aetros')

    level = 'INFO'
    fmt = '%(message)s' if format is None else format

    if debug is None:
        debug = is_debug()

    if debug:
        level = 'DEBUG'
        if format is None:
            fmt = coloredlogs.DEFAULT_LOG_FORMAT

    atty = None
    if '1' == os.getenv('AETROS_ATTY'):
        atty = True

    coloredlogs.install(fmt=fmt, level=level, logger=logger, isatty=atty)

    return logger


def ensure_docker_installed(logger, home_config=None):
    home_config = home_config or read_home_config()
    try:
        out = subprocess.check_output([home_config['docker'], '-v'])
        return True
    except Exception as e:
        logger.error('Docker is not installed: ' + (str(e)))
        sys.exit(2)


def docker_call(args, home_config=None):
    home_config = home_config or read_home_config()
    base = [home_config['docker']]
    return subprocess.check_output(base + args)


def loading_text(label='Loading ... '):
    import itertools, sys
    spinner = itertools.cycle(['-', '/', '|', '\\'])
    state = {'active': True}

    sys.stdout.write(label)
    sys.stdout.flush()

    def display_thread(state):
        try:
            while state['active']:
                sys.stdout.write(next(spinner))
                sys.stdout.flush()
                time.sleep(0.05)
                sys.stdout.write('\b')
        except (KeyboardInterrupt, SystemExit):
            return

    thread = None
    if not is_debug():
        thread = Thread(target=display_thread, args=[state])
        thread.daemon = True
        thread.start()

    def stop(done_label="done."):
        state['active'] = False
        thread and thread.join()
        sys.stdout.write(done_label + '\n')
        sys.stdout.flush()

    return stop


def get_option(dict, key, default=None, type=None):
    if key not in dict or dict[key] == '':
        return default

    if type == 'bool':
        bool(dict[key])

    return dict[key]


def extract_api_calls(line, callback, print_traceback=False, logger=None):
    failed_calls = []
    handled_calls = []
    filtered_line = line

    if hasattr(filtered_line, 'decode'):
        filtered_line = filtered_line.decode('utf-8')

    c = 0
    while True:
        start_pos = filtered_line.find('{aetros:')
        if -1 == start_pos:
            start_pos = filtered_line.find('{"aetros":')

        end_pos = filtered_line.find('}\n')
        eat_end = 2
        if -1 == end_pos:
            end_pos = filtered_line.find('}\r\n')
            eat_end = 3
            if -1 == end_pos:
                end_pos = filtered_line.find('}\r')
                eat_end = 2

        if start_pos == -1 or end_pos == -1:
            break

        line = filtered_line[start_pos:end_pos+1]

        try:
            call = yaml.load(line, Loader=yaml.RoundTripLoader)
            if callback(call) is False:
                failed_calls.append({'line': line, 'exception': Exception('Unknown API call.')})
            else:
                handled_calls.append(call)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if print_traceback:
                sys.__stderr__.write(traceback.format_exc())
            else:
                logger and logger.debug(traceback.format_exc())
            failed_calls.append({'line': line, 'exception': e})

        filtered_line = filtered_line[0:start_pos] + filtered_line[end_pos+eat_end:]

        c += 1

        if c > 10:
            # allow max 10 in one line
            break

    return handled_calls, filtered_line, failed_calls


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


class KeyNotConfiguredException(Exception):
    pass


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
        raise KeyNotConfiguredException("No SSH key configured for " + config['host']
                                        + ". See https://aetros.com/docu/trainer/authentication")

    key_description = key_filename if key_filename else 'from server'

    try:
        def lower_compression(self):
            import zlib
            # Use the default level of zlib compression
            self.z = zlib.compressobj(2)

        ZlibCompressor.__init__ = lower_compression

        ssh_stream.connect(config['host'], port=config['ssh_port'], key_filename=key_filename, username='git', compress=True, pkey=key)
        # ssh_stream.get_transport().window_size = 2147483647
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as e:
        if exit_on_failure:
            message = str(type(e).__name__) + ': ' + str(e)
            sys.stdout.write("Fatal: AETROS authentication against "+config['host']+" failed using key " +
                             key_description + ": " + message +
                             ". Did you setup SSH keys correctly? See https://aetros.com/docu/trainer/authentication\n")
            sys.exit(1)
        raise

    return ssh_stream


def setup_git_ssh(config):
    import tempfile
    ssh_command = config['ssh']
    ssh_command += ' -p ' + str(config['ssh_port'])
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
            sys.stderr.write('Error: could not load aetros home config at ' + os.path.realpath(path) + '\n')
            raise

        if custom_config is None:
            custom_config = {}

    config = {
        'host': 'trainer.aetros.com',
        'storage_dir': '~/.aetros',
        'ssh_key_base64': None,
        'image': None,
        'ssh': 'ssh',
        'git': 'git',
        'docker': 'docker',
        'docker_options': [],
        'http_port': 80,
        'https_port': 443,
        'ssh_port': 22,
        'ssl': True,
        'ssl_verify': True,
    }

    config.update(custom_config)

    if os.getenv('AETROS_HOST'):
        config['host'] = os.getenv('AETROS_HOST')

    if os.getenv('AETROS_STORAGE_DIR'):
        config['storage_dir'] = os.getenv('AETROS_STORAGE_DIR')

    if os.getenv('AETROS_SSH_KEY_BASE64'):
        config['ssh_key_base64'] = os.getenv('AETROS_SSH_KEY_BASE64')

    config['storage_dir'] = os.path.abspath(os.path.expanduser(config['storage_dir']))

    http = 'https://'
    host = config['host']

    if config['https_port'] != 443:
        host += ':' + str(config['https_port'])

    if not config['ssl']:
        http = 'http://'
        if config['http_port'] != 80:
            host += ':' + str(config['http_port'])

    config['url'] = http + host

    return config

ignore_pattern_cache = {}


def extract_parameters(full_definitions, overwritten = None, incoming_path = ''):
    container = {}

    if overwritten is None:
        overwritten = {}

    for parameter in full_definitions:
        param_type = parameter['type']
        param_subtype = parameter['subtype'] if 'subtype' in parameter else None

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
            if param_subtype == 'int':
                container[name] = int(value)
            else:
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
                if isinstance(v, (int)):
                    definition['subtype'] = 'int'

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
                regex = '^' + regex + '$'
            else:
                regex = '^.*' + regex + '$'

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


def git_local_job_ids(home_config, model):
    git_dir = os.path.normpath(home_config['storage_dir'] + '/' + model + '.git')

    output = subprocess.check_output([home_config['git'], '--bare', '--git-dir', git_dir, 'show-ref']).decode('utf-8')

    job_ids = []

    for line in output.split('\n'):
        if not ' ' in line:
            continue
        target, ref_name = line.split(' ')
        if ref_name.startswith('refs/aetros/job/'):
            job_ids.append(ref_name[len('refs/aetros/job/'):])

    return job_ids


def git_remote_job_ids(home_config, model):
    git_remote_url = 'git@%s:%s.git' % (home_config['host'], model)
    output = subprocess.check_output([home_config['git'], 'ls-remote', git_remote_url]).decode('utf-8')
    job_ids = []

    for line in output.split('\n'):
        if '\t' not in line:
            continue
        target, ref_name = line.split('\t')
        if ref_name.startswith('refs/aetros/job/'):
            job_ids.append(ref_name[len('refs/aetros/job/'):])

    return job_ids


def git_has_local_job(home_config, model, job_id):
    job_ids = git_local_job_ids(home_config, model)
    for full_job_id in job_ids:
        if full_job_id.startswith(job_id):
            return full_job_id


def git_has_remote_job(home_config, model, job_id):
    job_ids = git_remote_job_ids(home_config, model)
    for full_job_id in job_ids:
        if full_job_id.startswith(job_id):
            return full_job_id


def find_config(path = None, error_on_missing=False, return_default=True, logger=None):
    config = None

    if path:
        if os.path.exists(path):
            config = read_config(path, return_default=False, logger=logger)
            config['init_config_path'] = os.path.abspath(path)

    else:
        path = find_config_path()
        if path:
            config = read_config(path, return_default=False, logger=logger)
            config['init_config_path'] = os.path.abspath(path)

    if config:
        if 'import' in config and config['import']:
            inherited_config = find_config(os.path.dirname(os.path.abspath(path)) + '/' + config['import'])

            if inherited_config:
                inherited_config.update(config)
                config = inherited_config

        if config['model']:
            if 'init_config_path' in config:
                config['working_dir'] = os.path.relpath(os.path.dirname(config['init_config_path']), config['root'])
                config['configPath'] = os.path.relpath(config['init_config_path'], config['root'])
                del config['init_config_path']

            if return_default:
                return apply_config_defaults(config)
            else:
                return config

    if error_on_missing:
        sys.stderr.write('Error: No AETROS Trainer model name given. Specify it in aetros.yml '
                         '"model: user/model-name" or use "aetros init model-name".\n')
        sys.exit(2)

    return {'model': None}


def find_config_path(dir = None):
    if not dir:
        dir = os.path.abspath(os.getcwd())

    while True:
        if os.path.exists(dir+ '/aetros.yml'):
            return dir + '/aetros.yml'
        else:
            new_path = os.path.realpath(dir + '/..')
            if new_path == dir:
                # means we are at the end
                return None

            if new_path == os.path.realpath(os.path.expanduser('~')):
                # means we are at the home folder, where the home configuration is
                return None

            dir = new_path


def apply_config_defaults(config):
    defaults = {
        'model': None,
        'dockerfile': None,
        'command': None,
        'install': None,
        'ignore': None,
        'image': None,
        'server': None,
        'parameters': {},
        'import': None,
        'root': os.getcwd(),
        'working_dir': None,
        'servers': None,
        'configPath': None,
        'before_command': [],
    }

    defaults.update(config)

    return defaults


def read_config(path='aetros.yml', logger=None, return_default=True):
    path = os.path.normpath(os.path.abspath(os.path.expanduser(path)))

    config = {}

    if os.path.exists(path):
        f = open(path, 'r')

        config = yaml.load(f, Loader=yaml.RoundTripLoader)
        if config is None:
            config = {}

        if 'storage_dir' in config:
            del config['storage_dir']

        if 'model' in config and config['model']:
            config['root'] = os.path.dirname(path)

        logger and logger.debug('Config loaded from ' + os.path.realpath(path))

    if return_default:
        return apply_config_defaults(config)

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
