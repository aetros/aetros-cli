from __future__ import absolute_import
from __future__ import print_function

import sys
import aetros.const

command_summaries = [
    ['start', 'Starts a job of a model in current working directory'],
    ['predict', 'Runs a prediction locally'],
    ['upload-weights', 'Uploads weights as new or existing job.'],
    ['prediction-server', 'Spawns a http server that handles incoming data as input and predicts output.'],
    ['server', 'Spawns a job server that handles jobs managed through AETROS Trainer'],
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

    from aetros.commands.ServerCommand import ServerCommand
    from aetros.commands.UploadWeightsCommand import UploadWeightsCommand
    from aetros.commands.PredictCommand import PredictCommand
    from aetros.commands.PredictionServerCommand import PredictionServerCommand
    from aetros.commands.StartCommand import StartCommand

    commands_dict = {
        'start': StartCommand,
        'predict': PredictCommand,
        'upload-weights': UploadWeightsCommand,
        'prediction-server': PredictionServerCommand,
        'server': ServerCommand,
    }
    cmd_name, cmd_args = parseopts(args)

    if cmd_name not in commands_dict:
        print(("Command %s not found" % (cmd_name,)))
        sys.exit(1)

    command = commands_dict[cmd_name]()

    code = command.main(cmd_args)
    sys.exit(code)
