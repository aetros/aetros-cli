from __future__ import print_function, division

from __future__ import absolute_import
import logging
import os
import shutil
import subprocess
import sys
import traceback

import six

from aetros import api
from aetros.utils import read_home_config, unpack_full_job_id, read_config
from . import keras_model_utils
from .backend import JobBackend
from .Trainer import Trainer


class GitCommandException(Exception):
    cmd = None


def start(logger, full_id, hyperparameter=None, dataset_id=None, server='local', insights=False):
    """
    Starts the training process with all logging of a job_id
    :type id: string : job id or model name
    """
    id = None

    if full_id.count('/') == 1:
        owner, name = full_id.split('/')
    elif full_id.count('/') >= 2:
        owner, name, id = unpack_full_job_id(full_id)
    else:
        logger.error("Invalid id %s given. Supported formats: owner/modelName or owner/modelName/jobId." % (full_id, ))
        sys.exit(1)

    job_backend = JobBackend(model_name=owner + '/' + name)

    if id:
        job_backend.restart(id)
    else:
        try:
            create_info = api.create_job_info(full_id, hyperparameter, dataset_id)
        except api.ApiError as e:
            if 'Connection refused' in e.reason:
                logger.error("You are offline")
            logger.error("Can not start new job without knowing what model type it is. "
                         "Use your script directly if its a Python model.")
            raise

        if not create_info:
            raise Exception('Could not fetch model information. Are you online and have access to the given model?')

        job_backend.create(create_info=create_info, hyperparameter=hyperparameter, server=server, insights=insights)

    if not len(job_backend.get_job_model().config):
        raise Exception('Job does not have a configuration. Make sure you created the job via AETROS Trainer.')

    job_backend.start()

    if job_backend.is_simple_model():
        start_keras(logger, job_backend)
    else:
        start_custom(logger, job_backend)


def start_custom(logger, job_backend):
    job_model = job_backend.get_job_model()
    config = job_model.config

    work_tree = job_backend.git.work_tree

    my_env = os.environ.copy()
    if 'PYTHONPATH' not in my_env:
        my_env['PYTHONPATH'] = ''

    my_env['PYTHONPATH'] += ':' + os.getcwd()
    my_env['AETROS_MODEL_NAME'] = job_backend.model_name
    my_env['AETROS_JOB_ID'] = job_backend.job_id
    my_env['AETROS_ATTY'] = '1'
    my_env['AETROS_GIT'] = job_backend.git.get_base_command()

    if 'sourceGitDisabled' not in config or not config['sourceGitDisabled']:
        # "aetros run" sets this to true, so we don't check out (wrong) source
        # since at the job commit from "aetros run" we have already all necessary files attached.

        custom_git = False

        if 'gitCustom' in config and config['gitCustom']:
            custom_git = config['gitCustom']

        if custom_git and ('sourceGitUrl' not in config or not config['sourceGitUrl']):
            raise Exception('Server git url is not configured. Aborted')

        git_tree = 'master'

        if custom_git:
            git_url = config['sourceGitUrl']
        else:
            user_config = read_home_config()
            git_url = 'git@' + user_config['host'] + ':' + job_backend.model_name + '.git'

        if 'sourceGitTree' in config and config['sourceGitTree']:
            git_tree = config['sourceGitTree']

        logger.info("Clone job Git URL %s in %s" % (git_url, work_tree))
        logger.info("Using Git tree of '%s'" % (git_tree, ))

        try:
            if os.path.exists(work_tree):
                shutil.rmtree(work_tree)

            args = ['git', 'clone', git_url, work_tree]
            code = subprocess.call(args, stderr=sys.stderr, stdout=sys.stdout)
            if code != 0:
                raise Exception('Could not clone repository %s to %s' % (git_url, work_tree))

            # make sure the requested branch is existent in local git. Target FETCH_HEAD to this branch.
            git_execute(logger, work_tree, ['fetch', 'origin', git_tree])
            git_execute(logger, work_tree, ['reset', '--hard', 'FETCH_HEAD'])

            logger.info("Model source code checked out.")

        except GitCommandException as e:
            logger.error('Could not run "%s" for repository %s in %s. Look at previous output.' %(e.cmd, git_url, work_tree))
            raise

    project_config = read_config(work_tree + '/.aetros.yml')

    job_config = job_backend.job['config']

    if 'command' not in job_config and ('command' not in project_config or project_config['command'] == ''):
        logger.error('No "command" specified in .aetros.yml file. See Configuration section in the documentation.')
        sys.exit(1)

    if os.path.exists(work_tree + '/.aetros.yml'):
        job_backend.commit_file(work_tree + '/.aetros.yml', '.aetros.yml')

    if 'command' in job_config:
        command = job_config['command']
    else:
        command = project_config['command']

    image = None
    if 'image' in job_config:
        image = job_config['image']
    elif 'image' in project_config:
        image = project_config['image']

    logger.info("Switch working directory to " + work_tree)
    os.chdir(job_backend.git.work_tree)
    job_backend.detect_git_version()

    job_backend.set_system_info('command', str(command))

    if image is not None:
        if project_config['dockerfile'] or project_config['install']:
            dockerfile = project_config['dockerfile']

            if not (isinstance(dockerfile, six.string_types) and os.path.exists(dockerfile)):
                if isinstance(dockerfile, six.string_types):
                    dockerfile_content = dockerfile
                elif isinstance(dockerfile, list) and len(dockerfile) > 0:
                    if not dockerfile[0].startswith('FROM '):
                        dockerfile.insert('FROM ' + image, 0)

                    dockerfile_content = "\n".join(dockerfile)
                else:
                    dockerfile_content = 'FROM ' + image + '\nRUN '

                    if isinstance(project_config['install'], list):
                        dockerfile_content += '\n RUN '.join(project_config['install'])
                    else:
                        dockerfile_content += project_config['install']

                with open('Dockerfile', 'w') as f:
                    f.write(dockerfile_content)

                dockerfile = 'Dockerfile'
                job_backend.commit_file('Dockerfile')

            docker_build = [
                project_config['docker'],
                'build',
                '-t', job_backend.model_name,
                '-f', dockerfile,
                '.',
            ]

            logger.info("Prepare docker image: $ " + (' '.join(docker_build)))
            p = execute_command(args=docker_build, bufsize=1, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            if p.returncode:
                job_backend.crash('Image build error')
                sys.exit(p.returncode)

            image = job_backend.model_name

        job_backend.set_system_info('image', str(image))

        # make sure old container is removed
        subprocess.call([project_config['docker'], 'rm', job_backend.job_id])

        docker_command = [project_config['docker'], 'run', '--name', job_backend.job_id]
        docker_command += project_config['docker_options']
        docker_command += ['--mount', 'type=bind,source='+job_backend.git.work_tree+',destination=/exp']
        docker_command += ['-w', '/exp']
        docker_command.append(image)

        if isinstance(command, list):
            docker_command += command
        else:
            docker_command += ['sh', '-c', command]

        command = docker_command

    logger.warning("$ %s " % str(command))

    try:
        p = execute_command(args=command, bufsize=1, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        job_backend.set_system_info('exit_code', p.returncode)

        if p.returncode:
            job_backend.crash()

        sys.exit(p.returncode)

    except KeyboardInterrupt:
        logger.warning("Job aborted.")
        sys.exit(1)


def execute_command(**args):
    p = subprocess.Popen(**args)
    wait_stdout = sys.stdout.attach(p.stdout)
    wait_stderr = sys.stderr.attach(p.stderr)
    p.wait()

    wait_stdout()
    wait_stderr()

    return p


def git_execute(logger, repo_path, args):
    args = ['git', '--git-dir', repo_path + '/.git', '--work-tree', repo_path] + args
    logger.info("$ %s" % (' '.join(args), ))

    p = execute_command(args=args, bufsize=1, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    if p.returncode != 0:
        exception = GitCommandException("Git command returned not 0. " + (' '.join(args)))
        exception.cmd = (' '.join(args))
        raise exception


def start_keras(logger, job_backend):
    # we need to import keras here, so we know which backend is used (and whether GPU is used)
    os.chdir(job_backend.git.work_tree)

    # all our shapes are Tensorflow schema. (height, width, channels)
    import keras.backend
    if hasattr(keras.backend, 'set_image_dim_ordering'):
        keras.backend.set_image_dim_ordering('tf')

    if hasattr(keras.backend, 'set_image_data_format'):
        keras.backend.set_image_data_format('channels_last')

    from .KerasCallback import KerasCallback
    trainer = Trainer(job_backend)
    keras_logger = KerasCallback(job_backend, job_backend.logger)

    try:
        logger.info("Setup simple job")
        job_backend.progress(0, job_backend.job['config']['epochs'])

        logger.info("Start job")
        keras_model_utils.job_start(job_backend, trainer, keras_logger)

        job_backend.sync_weights()
        job_backend.done()

        logger.info("Done.")
        sys.exit(0)
    except KeyboardInterrupt:
        if job_backend.running:
            job_backend.set_status('ABORTED')

            if trainer.model:
                trainer.model.stop_training = True

            job_backend.sync_weights()
            job_backend.abort()
            logger.warning("Aborted.")

        sys.exit(1)
    except Exception as e:
        if trainer.model:
            trainer.model.stop_training = True

        logging.error(traceback.format_exc())
        job_backend.crash(e)
        raise
