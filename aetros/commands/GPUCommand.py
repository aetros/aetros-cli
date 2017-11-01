from __future__ import absolute_import
from __future__ import print_function
import argparse
import json

import six


class GPUCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.cuda_gpu

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' run')

        for gpu_id, gpu in six.iteritems(aetros.cuda_gpu.get_ordered_devices()):
            properties = aetros.cuda_gpu.get_device_properties(gpu['device'], all=True)
            print("GPU" + str(gpu_id) + ": " + str(properties['name']))
            print(json.dumps(properties, indent=4))
