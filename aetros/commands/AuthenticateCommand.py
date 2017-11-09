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

from aetros import api
from aetros.utils import read_home_config, get_ssh_key_for_host, create_ssh_stream


class AuthenticateCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' authenticate',
            description='Authenticates the machine with a new pair of SSH keys with a user account.')

        parsed_args = parser.parse_args(args)

        home_config = read_home_config()
        host = home_config['host']

        installed_key = get_ssh_key_for_host(host)
        key_exists_and_valid = False
        if installed_key:
            try:
                create_ssh_stream(home_config, exit_on_failure=False)
                key_exists_and_valid = True
            except Exception: pass

        if key_exists_and_valid:
            choice = six.moves.input("You have already configured a valid SSH (ssk_key: "+installed_key+") "
                                     "for "+host+".\nWant to create a new key? (y/N): ").lower()
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
        print("\n   https://" + host + "/confirm-machine/" + token)
        print("\nWaiting for confirmation ...")

        while True:
            time.sleep(3)
            response = api.http_request('machine-token/authorized?id=' + token, method='post')
            if response['status'] == 'confirmed':
                print("\n" + response['username'] + ' confirmed the public key. Test with "aetros id" or "ssh git@'+host+'".')
                private_key_path = os.path.expanduser('~/.ssh/aetros_' + response['username']+'_rsa')
                public_key_path = os.path.expanduser('~/.ssh/aetros_' + response['username']+'_rsa.pub')

                if not os.path.exists(os.path.dirname(private_key_path)):
                    os.makedirs(os.path.dirname(private_key_path))

                with open(private_key_path, 'w') as f:
                    f.write(ssh_key_private)

                with open(public_key_path, 'w') as f:
                    f.write(ssh_key_public)

                os.chmod(private_key_path, 0o600)
                os.chmod(public_key_path, 0o600)

                ssh_config_path = os.path.expanduser('~/.ssh/config')

                if not os.path.exists(os.path.dirname(ssh_config_path)):
                    os.makedirs(os.path.dirname(ssh_config_path))

                host_section = 'host '+host+'\n'
                identity_section = '    IdentityFile ~/.ssh/aetros_' + response['username']+'_rsa\n'

                if os.path.exists(ssh_config_path):
                    import re
                    regex = re.compile(r"^host\s+" + re.escape(host)+'\s*', re.IGNORECASE | re.MULTILINE)
                    with open(ssh_config_path, 'r+') as f:
                        config = f.read()

                        if regex.match(config):
                            config = regex.sub(host_section + identity_section, config, 1)
                        else:
                            config = host_section + identity_section + config

                        f.seek(0)
                        f.write(config)
                else:
                    with open(ssh_config_path, 'w') as f:
                        f.write(host_section + identity_section)

                print("Private key " + private_key_path + " installed in ~/.ssh/config for "+host+".\n")
                user = api.user()
                print("Key installed of account %s (%s)." % (user['username'], user['name']))
                sys.exit(0)
            if response['status'] == 'expired':
                print("Token expired.")
                sys.exit(1)


