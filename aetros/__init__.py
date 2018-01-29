from __future__ import absolute_import
from __future__ import print_function

import sys
import aetros.const

__version__ = const.__version__

command_summaries = [
    ['run', 'Starts a job from source code of the current directory.'],
    ['start', 'Starts a job from the model\'s remote Git repository.'],
    # ['predict', 'Runs a prediction locally'],
    # ['upload-weights', 'Uploads weights as new or existing job.'],
    # ['prediction-server', 'Spawns a http server that handles incoming data as input and predicts output.'],
    ['server', 'Connects the current machine as a cluster server with AETROS Trainer.'],
    ['api', 'Executes a API call through SSH connection at AETROS Trainer.'],
    ['model', 'Information about current model.'],
    ['jobs', 'List all job ids.'],
    ['job-push', 'Pushes a local job to AETROS Trainer.'],
    ['job-pull', 'Pulls a job to local Git repository.'],
    ['job-diff', 'Prints an unified diff of two jobs.'],
    ['job-checkout', 'Checks all files from a job out to a directory.'],
    ['job-files', 'List all files of a job.'],
    ['job-cat', 'Prints the content of a file in a job.'],
    ['job-commits', 'Lists all Git commits of a job'],
    ['home-config', 'Changes the global configuration in ~/aetros.yml.'],
    ['add', 'Adds a local file to a job tree.'],
    ['init', 'Creates a new model and places a aetros.yml in current working (or specified) directory.'],
    ['id', 'Shows under which account the machine is authenticated.'],
    ['gpu', 'Shows information about installed GPUs.'],
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
    from aetros.commands.ModelCommand import ModelCommand
    from aetros.commands.JobPushCommand import JobPushCommand
    from aetros.commands.JobPullCommand import JobPullCommand
    from aetros.commands.JobDiffCommand import JobDiffCommand
    from aetros.commands.JobCheckoutCommand import JobCheckoutCommand
    from aetros.commands.JobFilesCommand import JobFilesCommand
    from aetros.commands.JobCatCommand import JobCatCommand
    from aetros.commands.JobCommitsCommand import JobCommitsCommand
    from aetros.commands.HomeConfigCommand import HomeConfigCommand
    from aetros.commands.JobsCommand import JobsCommand
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
        'model': ModelCommand,
        'jobs': JobsCommand,
        'job-push': JobPushCommand,
        'job-pull': JobPullCommand,
        'job-diff': JobDiffCommand,
        'job-checkout': JobCheckoutCommand,
        'job-files': JobFilesCommand,
        'job-cat': JobCatCommand,
        'job-commits': JobCommitsCommand,
        'home-config': HomeConfigCommand,
        'add': AddCommand,
        'init': InitCommand,
        'gpu': GPUCommand,
    }

    if cmd_name not in commands_dict:
        print(("Command %s not found" % (cmd_name,)))
        sys.exit(1)

    from aetros.utils import get_logger

    import coloredlogs
    format = coloredlogs.DEFAULT_LOG_FORMAT if cmd_name == 'server' else None
    logger = get_logger('aetros-' + cmd_name, format=format)
    command = commands_dict[cmd_name](logger)

    code = command.main(cmd_args)
    sys.exit(code)
