from __future__ import absolute_import
from __future__ import print_function
import argparse

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

        print("CUDA version: " +str(aetros.cuda_gpu.get_version()))

        for gpu in aetros.cuda_gpu.get_ordered_devices():
            properties = aetros.cuda_gpu.get_device_properties(gpu['device'], all=True)
            free, total = aetros.cuda_gpu.get_memory(gpu['device'])
            print("%s GPU id=%s %s (memory %dGB, free %dGB)" %(gpu['fullId'], str(gpu['id']), properties['name'], total/1024/1024/1024, free/1024/1024/1024))
