import time

import datetime


def get_option(dict, key, default = None, type = None):
    if key not in dict:
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

    return st + '.' + str(ts%1)[2:6]

