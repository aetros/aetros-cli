from __future__ import print_function, division
from __future__ import absolute_import

import time
import json
import os
import subprocess
import sys
import six

from aetros.logger import GeneralLogger
from aetros.utils import unpack_full_job_id, read_home_config, flatten_parameters, get_ssh_key_for_host
from aetros.const import JOB_STATUS
from .backend import JobBackend
from .Trainer import Trainer

start_time = time.time()


class GitCommandException(Exception):
    cmd = None


def start(logger, full_id, fetch=True, env=None, volumes=None, gpu_devices=None):
    """
    Starts the training process with all logging of a job_id
    """

    owner, name, id = unpack_full_job_id(full_id)

    if isinstance(sys.stdout, GeneralLogger):
        # we don't want to have stuff written to stdout before in job's log
        sys.stdout.clear_buffer()

    job_backend = JobBackend(model_name=owner + '/' + name)

    if fetch:
        job_backend.fetch(id)

    job_backend.restart(id)
    job_backend.start()
    job_backend.set_status('PREPARE')
    job_backend.monitoring_thread.handle_max_time = False

    start_command(logger, job_backend, env, volumes, gpu_devices=gpu_devices)


def start_command(logger, job_backend, env=None, volumes=None, gpu_devices=None):
    work_tree = job_backend.git.work_tree
    home_config = read_home_config()

    if not env:
        env = {}

    if 'PYTHONPATH' not in env:
        env['PYTHONPATH'] = os.getenv('PYTHONPATH', '')

    env['PYTHONPATH'] += ':' + os.getcwd()
    env['AETROS_MODEL_NAME'] = job_backend.model_name
    env['AETROS_JOB_ID'] = str(job_backend.job_id)
    env['DEBUG'] = os.getenv('DEBUG', '')
    env['AETROS_ATTY'] = '1'
    env['AETROS_GIT'] = job_backend.git.get_base_command()

    if os.getenv('AETROS_SSH_KEY_BASE64'):
        env['AETROS_SSH_KEY_BASE64'] = os.getenv('AETROS_SSH_KEY_BASE64')
    elif get_ssh_key_for_host(home_config['host']):
        # we need to read the key into env so the docker container can connect to AETROS
        env['AETROS_SSH_KEY_BASE64'] = open(get_ssh_key_for_host(home_config['host']), 'r').read()

    job_config = job_backend.job['config']

    if 'command' not in job_config :
        job_backend.fail('No "command" given. See Configuration section in the documentation.')

    command = job_config['command']
    image = job_config['image']

    if job_backend.is_simple_model():
        if image:
            command = ['python']
        else:
            command = [sys.executable]
        command += ['-m', 'aetros', 'start-simple', job_backend.model_name + '/' + job_backend.job_id]

    # replace {{batch_size}} parameters
    if isinstance(job_config['parameters'], dict):
        for key, value in six.iteritems(flatten_parameters(job_config['parameters'])):
            if isinstance(command, list):
                for pos, v in enumerate(command):
                    if isinstance(command[pos], six.string_types):
                        command[pos] = command[pos].replace('{{' + key + '}}', json.dumps(value))
            else:
                command = command.replace('{{' + key + '}}', json.dumps(value))

    logger.info("Switch working directory to " + work_tree)
    os.chdir(job_backend.git.work_tree)

    docker_image_built = False
    if job_config['dockerfile'] or job_config['install']:
        dockerfile = job_config['dockerfile']
        if isinstance(dockerfile, six.string_types) and os.path.exists(dockerfile):
            pass
        else:
            if isinstance(dockerfile, six.string_types):
                dockerfile_content = dockerfile
            elif isinstance(dockerfile, list) and len(dockerfile) > 0:
                dockerfile_content = "\n".join(dockerfile)
            else:
                if image is None:
                    job_backend.fail("Image name missing, needed by `install` in aetros.yml")
                dockerfile_content = 'FROM ' + image + '\nRUN '

                if isinstance(job_config['install'], list):
                    dockerfile_content += '\n RUN '.join(job_config['install'])
                else:
                    dockerfile_content += job_config['install']

            dockerfile_content = '# CREATED BY AETROS because of "install" or "dockerfile" config in aetros.yml.\n' \
                                 + dockerfile_content

            with open('Dockerfile.aetros', 'w') as f:
                f.write(dockerfile_content)

            dockerfile = 'Dockerfile.aetros'
            job_backend.commit_file('Dockerfile.aetros')

        job_backend.set_system_info('image/dockerfile', dockerfile)
        docker_build = [
            home_config['docker'],
            'build',
            '-t', job_backend.model_name,
            '-f', dockerfile,
            '.',
        ]

        logger.info("Prepare docker image: $ " + (' '.join(docker_build)))
        job_backend.set_status('IMAGE BUILD')
        p = execute_command(args=docker_build, bufsize=1, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        if p.returncode:
            job_backend.fail('Image build error')
            sys.exit(p.returncode)

        docker_image_built = True
        image = job_backend.model_name

    docker_command = None
    if image:
        if not docker_image_built:
            logger.info("Pull docker image: $ " + image)
            job_backend.set_status('IMAGE PULL')
            execute_command(args=[home_config['docker'], 'pull', image], bufsize=1,
                stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        inspections = execute_command_stdout([home_config['docker'], 'inspect', image])
        inspections = json.loads(inspections.decode('utf-8'))
        if inspections:
            inspection = inspections[0]
            with job_backend.git.batch_commit('Docker image'):
                job_backend.set_system_info('image/id', inspection['Id'])
                job_backend.set_system_info('image/docker_version', inspection['DockerVersion'])
                job_backend.set_system_info('image/created', inspection['Created'])
                job_backend.set_system_info('image/container', inspection['Container'])
                job_backend.set_system_info('image/architecture', inspection['Architecture'])
                job_backend.set_system_info('image/os', inspection['Os'])
                job_backend.set_system_info('image/size', inspection['Size'])
                job_backend.set_system_info('image/rootfs', inspection['RootFS'])

        # make sure old container is removed
        subprocess.Popen([home_config['docker'], 'rm', job_backend.job_id], stderr=subprocess.PIPE).wait()

        docker_command = [home_config['docker'], 'run', '-t', '--name', job_backend.job_id]
        docker_command += home_config['docker_options']

        env['AETROS_GIT_WORK_DIR'] = '/job'
        docker_command += ['--mount', 'type=bind,source='+job_backend.git.work_tree+',destination=/job']

        env['AETROS_STORAGE_DIR'] = '/aetros'
        docker_command += ['--mount', 'type=bind,source='+job_backend.git.git_path+',destination='+'/aetros/' + job_backend.model_name + '.git']

        home_config_path = os.path.expanduser('~/aetros.yml')
        if os.path.exists(home_config_path):
            env['AETROS_HOME_CONFIG_FILE'] = '/aetros/aetros.yml'
            docker_command += ['--mount', 'type=bind,source='+home_config_path+',destination='+'/aetros/aetros.yml']

        docker_command += ['-w', '/job']

        # make sure the docker command receives all environment variables
        for k in six.iterkeys(env):
            docker_command += ['-e', k]

        if volumes:
            for volume in volumes:
                docker_command += ['-v', volume]

        if 'resources' in job_backend.job:
            assigned_resources = job_backend.job['resources']

            cpus = 1
            if 'cpu' in assigned_resources and assigned_resources['cpu']:
                cpus = assigned_resources['cpu']
            docker_command += ['--cpus', str(cpus)]

            memory = 1
            if 'memory' in assigned_resources and assigned_resources['memory']:
                memory = assigned_resources['memory']

            docker_command += ['--memory', str(memory * 1024 * 1024 * 1024)]

        if gpu_devices and (sys.platform == "linux" or sys.platform == "linux2"):
            # only supported on linux
            docker_command += ['--runtime', 'nvidia']
            docker_command += ['-e', 'NVIDIA_VISIBLE_DEVICES=' + (','.join(gpu_devices))]
            # support nvidia-docker1 as well
            # docker_command += ['--device', '/dev/nvidia1']

        docker_command.append(image)

        # since linux doesnt handle SIGINT when pid=1 process has no signal listener,
        # we need to make sure, we attached one to the pid=1 process
        trap = 'trapIt () { "$@"& pid="$!"; trap "kill -INT $pid" INT TERM; ' \
               'while kill -0 $pid > /dev/null 2>&1; do wait $pid; ec="$?"; done; exit $ec;};'

        if isinstance(command, list):
            command = ' '.join(command)

        docker_command += ['sh', '-c', trap + 'trapIt ' + command]
        command = docker_command

    job_backend.set_system_info('image/name', str(image))

    if not isinstance(command, list):
        command = ['sh', '-c', command]

    p = None
    exited = False
    wait_stdout = None
    wait_stderr = None
    try:
        job_backend.set_status('STARTED')
        logger.warning("$ %s " % (' '.join([json.dumps(a) for a in command])))

        command_env = os.environ.copy()
        command_env.update(env)

        # make sure maxTime limitation is correctly calculated
        job_backend.monitoring_thread.handle_max_time = True
        job_backend.monitoring_thread.handle_max_time_time = time.time()

        # Since JobBackend sends SIGINT to its current process group, it sends also to its parents when same pg.
        # We need to change the process group of the process, so this won't happen.
        # If we don't this, the master process receives the SIGINT as well.
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs['preexec_fn'] = os.setsid

        p = subprocess.Popen(args=command,
            bufsize=1, stderr=subprocess.PIPE, stdout=subprocess.PIPE,
            env=command_env, **kwargs)
        wait_stdout = sys.stdout.attach(p.stdout)
        wait_stderr = sys.stderr.attach(p.stderr)

        p.wait()
        wait_stdout()
        wait_stderr()

        exited = True

        sys.exit(p.returncode)
    except SystemExit:
        # We can not send a SIGINT to the child process
        # as we don't know whether it received it already (pressing CTRL+C) or not (sending SIGINT to this process only
        # instead of to the group), so we assume it received it. A second signal would force the exit.
        # sys.__stdout__.write("SystemExit with " + str(p.returncode) + ', exited: ' + str(exited) + ", early: "+str(job_backend.in_early_stop)+"\n")

        # make sure the process dies
        if docker_command:
            # docker run does not proxy INT signals to the docker-engine,
            # so we need to do it on our own directly.
            subprocess.Popen([home_config['docker'], 'kill', '--signal', 'INT', job_backend.job_id],
                stderr=subprocess.PIPE, stdout=subprocess.PIPE).wait()
            subprocess.Popen([home_config['docker'], 'wait', job_backend.job_id], stdout=subprocess.PIPE).wait()
        elif not exited and p and p.poll() is None:
            p.kill() # sends SIGINT
            p.wait()

        if exited:
            if p.returncode == 0:
                job_backend.stop(progress=JOB_STATUS.PROGRESS_STATUS_DONE)
            elif p.returncode == 1:
                job_backend.stop(progress=JOB_STATUS.PROGRESS_STATUS_ABORTED)
            else:
                job_backend.stop(progress=JOB_STATUS.PROGRESS_STATUS_FAILED)
        else:
            # master received SIGINT before the actual command exited.
            if not job_backend.in_early_stop:
                # master did not receive early_stop signal (maxTime limitation)
                # if not, the master received a stop signal by server or by hand (ctrl+c), so mark as aborted
                job_backend.abort()
            else:
                # let the on_shutdown listener handle the rest
                pass

                # only
                #     # check if there was a JobBackend in the command
                #     # if so, we do not add any further stuff to the git
                #     last_progress = job_backend.git.contents('aetros/job/status/progress.json')
                #     if last_progress is not None and int(last_progress) > 2:
                #         # make sure, we do not overwrite stuff written inside the sub-process by using our own
                #         # on_shutdown listener.
                #         job_backend.stop()


def execute_command_stdout(command, input=None):
    p = subprocess.Popen(command, bufsize=1, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = p.communicate(input)

    if p.returncode:
        sys.stderr.write(out)
        sys.stderr.write(err)
        raise Exception('Could not execute command: ' + str(command))

    return out


def execute_command(**kwargs):
    p = subprocess.Popen(**kwargs)
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
    if 'KERAS_BACKEND' not in os.environ:
        os.environ['KERAS_BACKEND'] = 'tensorflow'

    from . import keras_model_utils

    # we need to import keras here, so we know which backend is used (and whether GPU is used)
    os.chdir(job_backend.git.work_tree)
    logger.debug("Start simple model")

    # we use the source from the job commit directly
    with job_backend.git.batch_commit('Git Version'):
        job_backend.set_system_info('git_remote_url', job_backend.git.get_remote_url('origin'))
        job_backend.set_system_info('git_version', job_backend.git.job_id)

    # all our shapes are Tensorflow schema. (height, width, channels)
    import keras.backend
    if hasattr(keras.backend, 'set_image_dim_ordering'):
        keras.backend.set_image_dim_ordering('tf')

    if hasattr(keras.backend, 'set_image_data_format'):
        keras.backend.set_image_data_format('channels_last')

    from .KerasCallback import KerasCallback
    trainer = Trainer(job_backend)
    keras_logger = KerasCallback(job_backend, job_backend.logger)

    job_backend.progress(0, job_backend.job['config']['epochs'])

    logger.info("Start training")
    keras_model_utils.job_start(job_backend, trainer, keras_logger)

    job_backend.done()