from __future__ import division
from __future__ import absolute_import

import os
import time
import datetime
import numpy as np
import six
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
        'ssl_verify': True,
    }

    config.update(custom_config)

    config['storage_dir'] = os.path.abspath(os.path.expanduser(config['storage_dir']))

    return config


def read_config(path = '.aetros.yml', logger=None):
    path = os.path.normpath(os.path.expanduser(path))
    home_config = read_home_config(logger=logger)

    config = home_config
    if os.path.exists(path):
        f = open(path, 'r')

        custom_config = yaml.load(f)
        if custom_config is None:
            custom_config = {}

        config.update(custom_config)

        logger and logger.debug('Config loaded from ' + os.path.realpath(path))

    if 'parameters' not in config:
        config['parameters'] = {}

    config['storage_dir'] = os.path.normpath(os.path.expanduser(config['storage_dir']))

    return config

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
