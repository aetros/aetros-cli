from __future__ import absolute_import
from __future__ import print_function

import sys
import logging
import os

import aetros.const

__version__ = const.__version__

command_summaries = [
    ['start', 'Starts a job of a model in current working directory'],
    ['predict', 'Runs a prediction locally'],
    ['upload-weights', 'Uploads weights as new or existing job.'],
    ['prediction-server', 'Spawns a http server that handles incoming data as input and predicts output.'],
    ['server', 'Spawns a job server that handles jobs managed through AETROS Trainer.'],
    ['run', 'Executes a command on an AETROS server.'],
    ['api', 'Executes a API call through SSH connection.'],
    ['push-job', 'Pushes a local job to AETROS Trainer.'],
    ['pull-job', 'Pulls a job to local Git repository.'],
    ['add', 'Adds a local file to a job tree.'],
    ['init', 'Creates a new model and places a aetros.yml in current working directory pointing to this model.'],
    ['id', 'Shows under which account the machine is authenticated.'],
    ['gpu', 'Shows information about installed GPUs'],
]

def parseopts(args):
    if len(args) == 0:
        description = [''] + ['%-27s %s' % (i, j) for i, j in command_summaries]
        print("usage: aetros [command]")
        print("v%s\n" %(const.__version__))
        print(('Possible commands:\n' + (
            '\n'.join(description))))

        sys.exit(1)

    cmd_name = args[0]

    # all the args without the subcommand
    cmd_args = args[1:]

    return cmd_name, cmd_args


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    cmd_name, cmd_args = parseopts(args)
    if cmd_name == '--version':
        print(aetros.const.__version__)
        sys.exit(0)

    from aetros.commands.ApiCommand import ApiCommand
    from aetros.commands.PushJobCommand import PushJobCommand
    from aetros.commands.PullJobCommand import PullJobCommand
    from aetros.commands.ServerCommand import ServerCommand
    from aetros.commands.PredictCommand import PredictCommand
    from aetros.commands.PredictionServerCommand import PredictionServerCommand
    from aetros.commands.StartCommand import StartCommand
    from aetros.commands.StartSimpleCommand import StartSimpleCommand
    from aetros.commands.RunCommand import RunCommand
    from aetros.commands.AddCommand import AddCommand
    from aetros.commands.InitCommand import InitCommand
    from aetros.commands.IdCommand import IdCommand
    from aetros.commands.GPUCommand import GPUCommand
    from aetros.commands.AuthenticateCommand import AuthenticateCommand

    commands_dict = {
        'start': StartCommand,
        'start-simple': StartSimpleCommand,
        'authenticate': AuthenticateCommand,
        'predict': PredictCommand,
        'prediction-server': PredictionServerCommand,
        'server': ServerCommand,
        'run': RunCommand,
        'api': ApiCommand,
        'id': IdCommand,
        'push-job': PushJobCommand,
        'pull-job': PullJobCommand,
        'add': AddCommand,
        'init': InitCommand,
        'gpu': GPUCommand,
    }

    if cmd_name not in commands_dict:
        print(("Command %s not found" % (cmd_name,)))
        sys.exit(1)

    level = 'INFO'
    if '--debug' in args or os.getenv('DEBUG') == '1':
        level = 'DEBUG'

    atty = None
    if '1' == os.getenv('AETROS_ATTY'):
        atty = True

    import coloredlogs
    logger = logging.getLogger('aetros-'+cmd_name)
    coloredlogs.install(level=level, logger=logger, isatty=atty)
    command = commands_dict[cmd_name](logger)

    code = command.main(cmd_args)
    sys.exit(code)
