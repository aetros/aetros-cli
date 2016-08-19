from __future__ import division
from __future__ import absolute_import
import time
import datetime
import numpy as np


def get_option(dict, key, default=None, type=None):
    if key not in dict or dict[key] == '':
        return default

    if type == 'bool':
        bool(dict[key])

    return dict[key]


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
