from __future__ import absolute_import
from __future__ import print_function
import argparse
import os
import sys
import time
import yaml

from aetros.backend import BackendClient, EventListener, ApiClient


class RunClient(BackendClient):
    def configure(self, model_id):
        self.model_id = model_id

    def on_connect(self, reconnect=False):
        self.send_message({'register_run': self.api_key, 'model': self.model_id})
        messages = self.wait_for_at_least_one_message(self.s)

        if not messages:
            return False

        message = messages.pop(0)
        if isinstance(message, dict) and 'a' in message:
            if "ACCESS_DENIED" in message['a']:
                print("Access denied. Exiting")
                self.event_listener.fire('stop')
                self.active = False
                return False

            if "REGISTERED" in message['a']:
                self.registered = True
                print("Connected to %s:%d." % (self.api_host, self.api_port))
                self.event_listener.fire('registration')
                self.handle_messages(messages)
                return True

        print("Registration of job %s failed." % (self.job_id,))
        return False


class RunCommand:
    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.registered = False
        self.active = True

    def main(self, args):
        import aetros.const

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                         prog=aetros.const.__prog__ + ' run')
        parser.add_argument('--server', help="On which AETROS server the command should be executed. Default read in .aetros.yml")
        parser.add_argument('--model', help="Under which model this job should be listed. Default read in .aetros.yml")
        parser.add_argument('--host', help="Default trainer.aetros.com. Default read in ~/.aetros.yml.")
        parser.add_argument('--config', help="Default .aetros.yml in current working directory.")

        parsed_args = parser.parse_args(args)

        if not parsed_args.secure_key:
            parser.print_help()
            sys.exit()

        if parsed_args.show_stdout:
            self.show_stdout = True

        config_path = parsed_args.config if parsed_args.config else '.aetros.yml'
        config_path = os.path.normpath(config_path)
        config = {}

        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.load(file.read())

        event_listener = EventListener()

        model_id = None

        if config['model']:
            model_id = config['model']
        if parsed_args.model:
            model_id = parsed_args.model

        event_listener.on('registration', self.on_registration)
        event_listener.on('disconnect', self.on_disconnect)
        event_listener.on('stop', self.on_stop)

        self.client = RunClient(config, event_listener)
        self.client.configure(model_id)
        self.client.start()

        while self.active:
            if self.registered:
                pass

            time.sleep(1)

    def on_registration(self, params):
        self.registered = True

    def on_disconnect(self, params):
        self.registered = False

    def on_stop(self, params):
        self.active = False
