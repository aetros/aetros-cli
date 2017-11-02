from __future__ import absolute_import
from __future__ import print_function
import argparse
import hashlib
import socket

import os
import paramiko
import six
import sys

import time

from cryptography.hazmat.primitives import serialization
from ruamel import yaml

from aetros import api
from aetros.utils import read_home_config


class RegisterCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' run',
            description='Register the machine with a new pair of SSH keys.')

        parsed_args = parser.parse_args(args)

        home_config = read_home_config()

        if home_config['ssh_key']:
            choice = six.moves.input("You have already configured a SSH (ssk_key: "+home_config['ssh_key']+") "
                                     "for AETROS.\nWant to create a new key? (y/N): ").lower()
            if choice != 'y' and choice != 'yes':
                print("Aborted.")
                sys.exit(1)

        ssh_key = paramiko.RSAKey.generate(4096)
        ssh_key_private = ssh_key.key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()
        ).decode()
        ssh_key_public = 'rsa ' + ssh_key.get_base64()

        fingerprint = hashlib.md5(ssh_key.__str__()).hexdigest()
        fingerprint = ':'.join(a + b for a, b in zip(fingerprint[::2], fingerprint[1::2]))

        token = api.http_request('machine-token', None, {
            'host': socket.getfqdn(),
            'key': ssh_key_public
        })

        print("Open following link and login to confirm this machine's SSH key in your account.")
        print("Public Key Fingerprint: MD5:" + fingerprint)
        print("\n   https://" + home_config['host'] + "/confirm-machine/" + token)
        print("\nWaiting for confirmation ...")

        while True:
            time.sleep(3)
            response = api.http_request('machine-token/authorized?id=' + token, method='post')
            if response['status'] == 'confirmed':
                print(response['username'] + " confirmed the public key.")
                private_key_path = os.path.expanduser('~/.ssh/aetros_' + response['username']+'_rsa')
                public_key_path = os.path.expanduser('~/.ssh/aetros_' + response['username']+'_rsa.pub')

                if not os.path.exists(os.path.dirname(private_key_path)):
                    os.makedirs(os.path.dirname(private_key_path))

                with open(private_key_path, 'w') as f:
                    f.write(ssh_key_private)

                with open(public_key_path, 'w') as f:
                    f.write(ssh_key_public)

                with open(os.path.expanduser('~/aetros.yml'), 'r+') as f:
                    config = f.read()
                    config = config.replace('ssh_key', '#ssh_key')
                    config += '\nssh_key: ~/.ssh/aetros_' + response['username']+'_rsa'
                    f.seek(0)
                    f.write(config)

                sys.exit(0)
            if response['status'] == 'expired':
                print("Token expired.")
                sys.exit(1)


