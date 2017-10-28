from __future__ import absolute_import
from __future__ import print_function
import argparse
import json


class GPUCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const
        import aetros.cuda_gpu

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' run')

        for i in range(0, aetros.cuda_gpu.get_installed_devices()):
            properties = aetros.cuda_gpu.get_device_properties(i, all=True)
            print("GPU" + str(i) + ": " + str(properties['name']))
            print(json.dumps(properties, indent=4))
