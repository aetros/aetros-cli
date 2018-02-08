from __future__ import print_function, division
from __future__ import absolute_import

import hashlib
import re
import shutil
import time
import simplejson
import os
import subprocess
import sys
import psutil
import signal
import six

from aetros.logger import GeneralLogger
from aetros.utils import unpack_full_job_id, read_home_config, flatten_parameters, get_ssh_key_for_host, \
    extract_api_calls, is_debug
from aetros.const import JOB_STATUS, __version__
from .backend import JobBackend
from .Trainer import Trainer

class GitCommandException(Exception):
    cmd = None


def start(logger, full_id, fetch=True, env=None, volumes=None, cpus=None, memory=None, gpu_devices=None, offline=False):
    """
    Starts the job with all logging of a job_id
    """

    owner, name, id = unpack_full_job_id(full_id)

    if isinstance(sys.stdout, GeneralLogger):
        # we don't want to have stuff written to stdout before in job's log
        sys.stdout.clear_buffer()

    job_backend = JobBackend(model_name=owner + '/' + name)

    if fetch:
        job_backend.fetch(id)

    job_backend.restart(id)
    job_backend.start(collect_system=False, offline=offline)
    job_backend.set_status('PREPARE', add_section=False)

    job = job_backend.get_job_model()

    if not cpus:
        cpus = job.get_cpu()

    if not memory:
        memory = job.get_memory()

    if not gpu_devices and job.get_gpu():
        # if requested 2 GPUs and we have 3 GPUs with id [0,1,2], gpus should be [0,1]
        gpu_devices = []
        for i in range(0, job.get_gpu()):
            gpu_devices.append(i)

    start_command(logger, job_backend, env, volumes, cpus=cpus, memory=memory, gpu_devices=gpu_devices, offline=offline)


def start_command(logger, job_backend, env_overwrite=None, volumes=None, cpus=1, memory=1,
                  gpu_devices=None, offline=False):

    home_config = read_home_config()

    env = {}
    if env_overwrite:
        env.update(env_overwrite)

    start_time = time.time()
    env['AETROS_MODEL_NAME'] = job_backend.model_name
    env['AETROS_JOB_ID'] = str(job_backend.job_id)
    env['AETROS_OFFLINE'] = '1' if offline else ''
    env['AETROS_GIT_INDEX_FILE'] = job_backend.git.index_path
    env['DEBUG'] = os.getenv('DEBUG', '')
    env['PYTHONUNBUFFERED'] = os.getenv('PYTHONUNBUFFERED', '1')
    env['AETROS_ATTY'] = '1'
    env['AETROS_GIT'] = job_backend.git.get_base_command()

    env['PATH'] = os.getenv('PATH', '')
    if 'PYTHONPATH' not in env:
        env['PYTHONPATH'] = os.getenv('PYTHONPATH', '')

    if os.getenv('AETROS_SSH_KEY_BASE64'):
        env['AETROS_SSH_KEY_BASE64'] = os.getenv('AETROS_SSH_KEY_BASE64')
    elif get_ssh_key_for_host(home_config['host']):
        # we need to read the key into env so the docker container can connect to AETROS
        env['AETROS_SSH_KEY_BASE64'] = open(get_ssh_key_for_host(home_config['host']), 'r').read()

    job_config = job_backend.job['config']

    if 'command' not in job_config:
        job_backend.fail('No "command" given. See Configuration section in the documentation.')

    job_commands = job_config['command']
    docker_image = job_config['image']

    if job_backend.is_simple_model():
        if docker_image:
            simple_command = ['python']
        else:
            simple_command = [sys.executable]

        simple_command += ['-m', 'aetros', 'start-simple', job_backend.model_name + '/' + job_backend.job_id]
        job_commands = {'run': ' '.join(simple_command)}

    if job_commands is None:
        raise Exception('No command specified.')

    if not isinstance(job_commands, list) and not isinstance(job_commands, dict):
        job_commands = [job_commands]

    # replace {{batch_size}} parameters
    if isinstance(job_config['parameters'], dict):
        for key, value in six.iteritems(flatten_parameters(job_config['parameters'])):
            if isinstance(job_commands, list):
                for k, v in enumerate(job_commands):
                    if isinstance(job_commands[k], six.string_types):
                        job_commands[k] = job_commands[k].replace('{{' + key + '}}', simplejson.dumps(value))

            elif isinstance(job_commands, dict):
                for k, v in six.iteritems(job_commands):
                    if isinstance(job_commands[k], six.string_types):
                        job_commands[k] = job_commands[k].replace('{{' + key + '}}', simplejson.dumps(value))

    job_backend.set_system_info('commands', job_commands)
    os.chdir(job_backend.git.work_tree)

    docker_image_built = False

    if docker_image and (job_config['dockerfile'] or job_config['install']):
        rebuild_image = job_config['rebuild_image'] if 'rebuild_image' in job_config else False
        docker_image = docker_build_image(logger, home_config, job_backend, rebuild_image)
        docker_image_built = True

    job_backend.collect_device_information(gpu_devices)

    state = {'last_process': None}
    job_backend.set_system_info('processRunning', False, True)

    def pause():
        if not state['last_process'] or state['last_process'].poll() is not None:
            # no running process
            return

        if docker_image:
            if docker_pause(logger, home_config, job_backend):
                job_backend.set_paused(True)
        else:
            os.killpg(os.getpgid(state['last_process'].pid), signal.SIGSTOP)
            job_backend.set_paused(True)

    def cont():
        if not state['last_process'] or state['last_process'].poll() is not None:
            # no running process
            return

        job_backend.set_paused(False)
        if docker_image:
            docker_continue(logger, home_config, job_backend)
        else:
            os.killpg(os.getpgid(state['last_process'].pid), signal.SIGCONT)

    job_backend.on_pause = pause
    job_backend.on_continue = cont

    if docker_image:
        env['AETROS_GIT_INDEX_FILE'] = '/aetros/' + job_backend.model_name + '.git/' + os.path.basename(env['AETROS_GIT_INDEX_FILE'])

        with job_backend.git.batch_commit('JOB_SYSTEM_INFORMATION'):
            aetros_environment = {'aetros_version': __version__, 'variables': env.copy()}
            if 'AETROS_SSH_KEY' in aetros_environment['variables']: del aetros_environment['variables']['AETROS_SSH_KEY']
            if 'AETROS_SSH_KEY_BASE64' in aetros_environment['variables']: del aetros_environment['variables']['AETROS_SSH_KEY_BASE64']
            job_backend.set_system_info('environment', aetros_environment)

            job_backend.set_system_info('memory_total', memory * 1024 * 1024 * 1024)

            import cpuinfo
            cpu = cpuinfo.get_cpu_info()
            job_backend.set_system_info('cpu_name', cpu['brand'])
            job_backend.set_system_info('cpu', [cpu['hz_actual_raw'][0], cpus])

        job_backend.start_monitoring(cpu_cores=cpus, gpu_devices=gpu_devices, docker_container=job_backend.job_id)

        if not docker_image_built:
            docker_pull_image(logger, home_config, job_backend)

        docker_image_information(logger, home_config, job_backend)

        # make sure old container is removed
        subprocess.Popen([home_config['docker'], 'rm', job_backend.job_id], stderr=subprocess.PIPE).wait()

        command = docker_command_wrapper(logger, home_config, job_backend, volumes, cpus, memory, gpu_devices, env)

        # since linux doesnt handle SIGINT when pid=1 process has no signal listener,
        # we need to make sure, we attached one to the pid=1 process
        trap = 'trapIt () { "$@"& pid="$!"; trap "kill -INT $pid" INT TERM; ' \
               'while kill -0 $pid > /dev/null 2>&1; do wait $pid; ec="$?"; done; exit $ec;};'

        command.append(docker_image)
        command += ['/bin/sh', '-c', trap + 'trapIt /bin/sh /job/aetros/command.sh']
    else:
        # non-docker
        # env['PYTHONPATH'] += ':' + os.getcwd()
        job_backend.collect_system_information()
        job_backend.collect_environment(env)
        job_backend.start_monitoring(gpu_devices=gpu_devices)

        command = ['/bin/sh', job_backend.git.work_tree + '/aetros/command.sh']

    logger.debug("$ %s " % (' '.join([simplejson.dumps(a) for a in command])))
    job_backend.set_system_info('image/name', str(docker_image))

    p = None
    exited = False
    last_return_code = None
    state['last_process'] = None
    all_done = False
    command_stats = None
    files = job_backend.file_list()

    def clean():
        # clear working tree
        shutil.rmtree(job_backend.git.work_tree)

    def on_force_exit():
        # make sure the process dies
        clean()

        with open(os.devnull, 'r+b', 0) as DEVNULL:
            if docker_image:
                # docker run does not proxy INT signals to the docker-engine,
                # so we need to do it on our own directly.
                subprocess.Popen(args=[home_config['docker'], 'kill', job_backend.job_id], stdout=DEVNULL, stderr=DEVNULL).wait()
            elif not exited and state['last_process'] and state['last_process'].poll() is None:
                # wait for last command
                os.killpg(os.getpgid(state['last_process'].pid), signal.SIGKILL)

    job_backend.on_force_exit = on_force_exit

    try:
        job_backend.set_status('STARTED', add_section=False)
        # logger.warning("$ %s " % (str(command),))

        # make sure maxTime limitation is correctly calculated
        job_backend.monitoring_thread.handle_max_time = True
        job_backend.monitoring_thread.handle_max_time_time = time.time()

        # Since JobBackend sends SIGINT to its current process group, it sends also to its parents when same pg.
        # We need to change the process group of the process, so this won't happen.
        # If we don't this, the master process (server command e.g.) receives the SIGINT as well.
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs['preexec_fn'] = os.setsid

        # only use full env when no image used

        command_env = env
        if not docker_image:
            command_env = os.environ.copy()
            command_env.update(env)
            if os.environ.get('LD_LIBRARY_PATH', None):
                command_env['LD_LIBRARY_PATH_ORI'] = command_env['LD_LIBRARY_PATH']

        def write_command_sh(job_command):
            f = open(job_backend.git.work_tree + '/aetros/command.sh', 'w+')

            if not docker_image:
                # new shells unset LD_LIBRARY_PATH automatically, so we make sure it will be there again
                f.write('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_ORI;\n')

            if 'working_dir' in job_config and job_config['working_dir']:
                f.write('cd %s;\n' % (job_config['working_dir'],))

            f.write(job_command)
            f.close()

        def read_line(line):
            handled, filtered_line, failed = extract_api_calls(line, job_backend.handle_stdout_api, logger=logger)

            if is_debug():
                for call in handled:
                    logger.debug('STDOUT API CALL: ' + str(call))

            for fail in failed:
                logger.warning("API call failed '%s': %s %s"
                               % (str(fail['line']), str(type(fail['exception']).__name__), str(fail['exception'])))

            return filtered_line

        def exec_command(id, command, job_command):
            write_command_sh(job_command)
            print('$ ' + job_command.strip() + '\n')
            args = command
            logger.debug('$ ' + ' '.join([simplejson.dumps(a) for a in args]))
            state['last_process'] = subprocess.Popen(
                args=args, bufsize=1, stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=command_env, **kwargs
            )
            job_backend.set_system_info('processRunning', True, True)
            wait_stdout = sys.stdout.attach(state['last_process'].stdout, read_line=read_line)
            wait_stderr = sys.stderr.attach(state['last_process'].stderr)
            state['last_process'].wait()
            job_backend.set_system_info('processRunning', True, False)
            wait_stdout()
            wait_stderr()
            # make sure a new line is printed after a command
            print("")
            return state['last_process']

        done = 0
        total = len(job_commands)
        job_backend.set_system_info('command_stats', command_stats, True)
        if isinstance(job_commands, list):
            command_stats = [{'rc': None, 'started': None, 'ended': None} for x in job_commands]
            for k, job_command in enumerate(job_commands):
                job_backend.set_status('Command ' + str(k+1))

                command_stats[k]['started'] = time.time() - start_time
                job_backend.set_system_info('command_stats', command_stats, True)

                command_env['AETROS_JOB_NAME'] = 'command_' + str(k)
                p = exec_command(k, command, job_command)
                last_return_code = p.poll()

                command_stats[k]['rc'] = last_return_code
                command_stats[k]['ended'] = time.time() - start_time
                job_backend.set_system_info('command_stats', command_stats, True)

                if last_return_code == 0:
                    done += 1
                else:
                    # one failed, so exit and don't execute next
                    break

        if isinstance(job_commands, dict):
            command_stats = {}
            for name, job_command in six.iteritems(job_commands):
                command_stats[name] = {'rc': None, 'started': None, 'ended': None}

            for name, job_command in six.iteritems(job_commands):
                job_backend.set_status('Command ' + name)

                command_stats[name]['started'] = time.time() - start_time
                job_backend.set_system_info('command_stats', command_stats, True)

                # important to prefix it, otherwise name='master' would reset all stats in controller backend
                command_env['AETROS_JOB_NAME'] = 'command_' + name
                p = exec_command(name, command, job_command)
                last_return_code = p.poll()

                command_stats[name]['rc'] = last_return_code
                command_stats[name]['ended'] = time.time() - start_time
                job_backend.set_system_info('command_stats', command_stats, True)

                if last_return_code == 0:
                    done += 1
                else:
                    # one failed, so exit and don't execute next
                    break

        all_done = done == total
        exited = True

        if state['last_process']:
            sys.exit(state['last_process'].poll())
        else:
            sys.exit(1)

    except SystemExit:
        # since we started the command in a new process group, a SIGINT or CTRL+C on this process won't affect
        # our actual command process. So we need to take care that we stop everything.
        logger.debug("SystemExit, exited=%s, all-done=%s, has-last-process=%s, pid=%s" %(
            str(exited),
            str(all_done),
            state['last_process'] is not None,
            state['last_process'].poll() if state['last_process'] else None
        ))

        # make sure the process dies
        if docker_image:
            # docker run does not proxy INT signals to the docker-engine,
            # so we need to do it on our own directly.
            p = subprocess.Popen(args=[home_config['docker'], 'inspect', job_backend.job_id],
                stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            p.wait()
            if p.poll() == 0:
                subprocess.Popen(args=[home_config['docker'], 'stop', '-t', '5', job_backend.job_id]).wait()
        elif not exited and state['last_process'] and state['last_process'].poll() is None:
            # wait for last command
            os.killpg(os.getpgid(state['last_process'].pid), signal.SIGINT)
            state['last_process'].wait()

        if 'output' in job_config and job_config['output']:
            upload_output_files(job_backend, job_config['output'])

        if exited:
            if all_done:
                job_backend.stop(progress=JOB_STATUS.PROGRESS_STATUS_DONE)
            else:
                job_backend.stop(progress=JOB_STATUS.PROGRESS_STATUS_FAILED)
        else:
            # master received SIGINT before the all job commands exited.
            if not job_backend.in_early_stop:
                # in_early_stop indicates whether we want to have a planned stop (maxTime limitation for example),
                # which should mark the job as done, not as abort().
                # if this is not set, we the master received a SIGINT without early_stop, so mark as aborted.
                job_backend.abort()
            else:
                # let the on_shutdown listener handle the rest
                pass

        clean()


def upload_output_files(job_backend, files):
    if not files:
        return

    if isinstance(files, six.string_types):
        files = [files]

    job_backend.set_status('UPLOAD JOB DATA')

    for file in files:
        path = job_backend.git.work_tree + '/' + file
        if os.path.exists(path):
            job_backend.git.add_file_path(file, job_backend.git.work_tree, verbose=True)
        else:
            print("Warning: Job output file %s does not exist." % (file, ))

    job_backend.git.commit_index('UPLOAD JOB DATA')


def docker_pull_image(logger, home_config, job_backend):
    image = job_backend.job['config']['image']

    job_backend.set_status('IMAGE PULL')
    logger.info("$ docker pull " + image)

    execute_command(args=[home_config['docker'], 'pull', image], bufsize=1, stderr=subprocess.PIPE,
        stdout=subprocess.PIPE)


def docker_image_information(logger, home_config, job_backend):
    image = job_backend.job['config']['image']

    inspections = execute_command_stdout([home_config['docker'], 'inspect', image])
    inspections = simplejson.loads(inspections.decode('utf-8'))

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
            if 'RootFS' in inspection:
                job_backend.set_system_info('image/rootfs', inspection['RootFS'])


def docker_pause(logger, home_config, job_backend):
    docker_command = [home_config['docker'], 'pause', job_backend.job_id]

    p = subprocess.Popen(docker_command)
    p.wait()

    return True if p.poll() == 0 else False


def docker_continue(logger, home_config, job_backend):
    docker_command = [home_config['docker'], 'unpause', job_backend.job_id]

    p = subprocess.Popen(docker_command)
    p.wait()

    return True if p.poll() == 0 else False


def docker_send_signal(logger, home_config, job_backend, signal):
    docker_command = [home_config['docker'], 'kill', '-s', signal, job_backend.job_id]
    execute_command_stdout(docker_command)


def docker_command_wrapper(logger, home_config, job_backend, volumes, cpus, memory, gpu_devices, env):
    docker_command = [home_config['docker'], 'run', '-t', '--rm', '--name', job_backend.job_id]
    docker_command += home_config['docker_options']

    env['AETROS_GIT_WORK_DIR'] = '/job'
    docker_command += ['--mount', 'type=bind,source=' + job_backend.git.work_tree + ',destination=/job']

    if not os.path.exists(job_backend.git.work_tree + '/aetros/'):
        os.makedirs(job_backend.git.work_tree + '/aetros/')

    env['AETROS_STORAGE_DIR'] = '/aetros'
    docker_command += ['--mount',
        'type=bind,source=' + job_backend.git.git_path + ',destination=' + '/aetros/' + job_backend.model_name + '.git']

    home_config_path = os.path.expanduser('~/aetros.yml')
    if os.path.exists(home_config_path):
        env['AETROS_HOME_CONFIG_FILE'] = '/aetros/aetros.yml'
        docker_command += ['--mount', 'type=bind,source=' + home_config_path + ',destination=' + '/aetros/aetros.yml']

    docker_command += ['-w', '/job']

    # following makes no sense to pass to Docker
    env_blacklist = ['PATH', 'PYTHONPATH']

    # make sure the docker command receives all environment variables
    for k in six.iterkeys(env):
        if k in env_blacklist:
            continue
        docker_command += ['-e', k]

    docker_command += ['-e', 'AETROS_JOB_NAME']

    if volumes:
        for volume in volumes:
            docker_command += ['-v', volume]

    docker_command += ['--cpus', str(cpus)]
    docker_command += ['--memory', str(memory * 1024 * 1024 * 1024)]

    if gpu_devices and (sys.platform == "linux" or sys.platform == "linux2"):
        # only supported on linux
        docker_command += ['--runtime', 'nvidia']
        docker_command += ['-e', 'NVIDIA_VISIBLE_DEVICES=' + (','.join(map(str,gpu_devices)))]
        # support nvidia-docker1 as well
        # docker_command += ['--device', '/dev/nvidia1']

    return docker_command


def docker_build_image(logger, home_config, job_backend, rebuild_image=False):
    job_config = job_backend.job['config']
    image = job_config['image']
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

        dockerfile_content = '# CREATED BY AETROS because of "install" or "dockerfile" config in aetros.yml.\n' + dockerfile_content

        with open('Dockerfile.aetros', 'w') as f:
            f.write(dockerfile_content)

        dockerfile = 'Dockerfile.aetros'
        job_backend.commit_file('Dockerfile.aetros')

    job_backend.set_system_info('image/dockerfile', dockerfile)

    image = job_backend.model_name.lower()

    if 'category' in job_config and job_config['category']:
        image += '_' + job_config['category'].lower()

    if 'configPath' in job_config and job_config['configPath']:
        config_path = job_config['configPath']\
            .replace('aetros.yml', '')\
            .replace('aetros-', '')\
            .replace('.yml', '')

        if config_path:
            image += '_' + config_path.lower()

    image = image.strip('/')
    image = re.sub('[^A-Z_/\-a-z0-9]+', '-', image).strip('-')

    m = hashlib.md5()
    m.update(simplejson.dumps([job_config['image'], job_config['install']]).encode('utf-8'))
    image += ':' + m.hexdigest()[0:9]

    docker_build = [home_config['docker'], 'build']

    if rebuild_image:
        docker_build += ['--no-cache']

    docker_build += ['-t', image, '-f', dockerfile, '.', ]

    job_backend.set_status('IMAGE BUILD')
    logger.info("Prepare docker image: $ " + (' '.join(docker_build)))
    p = execute_command(args=docker_build, bufsize=1, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    if p.returncode:
        job_backend.fail('Image build error')
        sys.exit(p.returncode)

    return image


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
    logger.info("$ %s" % (' '.join(args),))

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
