from __future__ import division
from __future__ import absolute_import

import os
import re
import time
import datetime

import numpy as np
import signal
import six
import sys
import yaml


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


def read_home_config(path = '~/.aetros.yml', logger=None):
    path = os.path.normpath(os.path.expanduser(path))
    custom_config = {}

    if os.path.exists(path):
        f = open(path, 'r')
        try:
            logger and logger.debug('Home config loaded from ' + os.path.realpath(path))
            custom_config = yaml.load(f)
        except:
            raise Exception('Could not load aetros home config at ' + os.path.realpath(path))

        if custom_config is None:
            custom_config = {}

    config = {
        'host': os.getenv('API_HOST') or 'trainer.aetros.com',
        'storage_dir': '~/.aetros',
        'ssh_key': os.getenv('AETROS_SSH_KEY'),
        'ssh': 'ssh',
        'image': None,
        'docker': 'docker',
        'docker_options': [],
        'ssl_verify': True,
    }

    config.update(custom_config)

    config['storage_dir'] = os.path.abspath(os.path.expanduser(config['storage_dir']))

    return config

ignore_pattern_cache = {}


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


def read_config(path = '.aetros.yml', logger=None):
    path = os.path.normpath(os.path.expanduser(path))
    home_config = read_home_config(logger=logger)

    config = {
        'dockerfile': None,
        'command': None,
        'install': None,
        'ignore': None,
        'server': None,
        'parameters': {},
        'servers': None,
        'before_command': [],
    }

    config.update(home_config)
    if os.path.exists(path):
        f = open(path, 'r')

        custom_config = yaml.load(f)
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
