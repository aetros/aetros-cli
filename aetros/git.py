import traceback

import simplejson
import os
import shutil
import subprocess

import six
from threading import Thread, Lock
import time
import sys

from aetros.utils import invalid_json_values, setup_git_ssh, create_ssh_stream, read_home_config, is_debug2


class GitCommandException(Exception):
    pass


class Git:
    """
    This class is used to store and sync all job data to local git or (if online) stream files directly to AETROS Trainer server.
    
    Git.stream_file and Git.store_file both stream new data directly to the server. At the end (Git.end) we commit
    the files locally and (if online) store the blob on the server's git. On a `git push` git detects that the content
    is already on our server which eliminates useless content transmissions.
    
    In either way (online or offline) all job information is stored in the local git repository and can be pushed any
    time to the AETROS git server. If the training happened in offline modus, one can push the job's ref (e.g. git push origin refs/aetros/job/<id>)
    to make the job available in AETROS Trainer later on.
    
    Job id is created always in the local git, except the job has been created through the AETROS Trainer interface.
    If created in AETROS Trainer, we retrieve the initial configuration of the job using `git pull origin refs/aetros/job/<id>` and read
    its `aetros/job.json` blob of the head tree.
    """
    def __init__(self, logger, client, config, model_name, is_master):
        self.logger = logger
        self.client = client

        self.config = config
        self.git_host = config['host']
        self.storage_dir = config['storage_dir']

        self.model_name = model_name
        self.is_master = is_master

        self.git_path = os.path.normpath(self.storage_dir + '/' + model_name + '.git')

        self.command_lock = Lock()
        self.stream_files_lock = Lock()
        self.index_lock = Lock()
        self.push_lock = Lock()

        self.debug = False
        self.last_push_time = 0
        self.active_push = False
        self.index_path = None

        self.job_id = None
        self.active_thread = False
        self.thread_push_instance = None

        self.synced_object_shas = {}

        self.git_batch_commit = False

        self.git_batch_commit_messages = []

        self.keep_stream_files = False

        self.streamed_files = {}
        self.store_files = {}

        git_not_found = 'Git binary not available. Please install Git >= 2.3.0 first and make it available in $PATH.'
        try:
            if subprocess.Popen(['git', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).wait() > 0:
                self.logger.error(git_not_found)
                sys.exit(2)
        except OSError:
            self.logger.error(git_not_found)
            sys.exit(2)

        ssh_not_found = 'SSH binary not available. Please install SSH first and make it available in $PATH.'
        try:
            if subprocess.Popen(['ssh', '-V'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).wait() > 0:
                self.logger.error(ssh_not_found)
                sys.exit(2)
        except OSError:
            self.logger.error(ssh_not_found)
            sys.exit(2)

        self.delete_git_ssh = setup_git_ssh(config)
        self.logger.debug("GIT_SSH='" + str(os.getenv('GIT_SSH'))+"'")
        self.git_name = None
        self.git_email = None

        import getpass
        self.git_name = getpass.getuser()
        import socket
        self.git_email = self.git_name + '@' + socket.gethostname()

        # check if its a git repo
        if os.path.exists(self.git_path):
            out, code, err = self.command_exec(['remote'])
            if code != 0:
                raise Exception('Given git_path (%s) already exists and does not seem to be a git repository. Error: %s' % (self.git_path, err))
        else:
            os.makedirs(self.git_path)
            self.command_exec(['init'])
            self.command_exec(['remote', 'add', 'origin', self.git_url])

        # make sure its not called before git init
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        # requires the temp folder
        self.prepare_index_file()

        # check if given repo_path is current folder.
        # check its origin remote and see if model_name matches
        self.origin_url = self.get_remote_url('origin')

        if self.origin_url and self.git_url not in self.origin_url:
            logger.warning("It seems you switched between aetros.com and an on-premise installation or updated host " \
                           "in your home configuration Given git_path (%s) points to a repository (%s) that is not "
                           "the git repo of the model (%s). " \
                           "We updated the remote origin automatically." \
                            % (self.git_path, self.origin_url, self.git_url))

            self.command_exec(['remote', 'remove', 'origin'])
            self.command_exec(['remote', 'add', 'origin', self.git_url])


    def get_remote_url(self, origin_name):
        output = self.command_exec(['remote', '-v'], allowed_to_fail=True)[0].decode('utf-8').strip().split('\n')

        for line in output:
            if line.startswith(origin_name):
                return line[len(origin_name)+1:line.index('.git')+4]

    @property
    def work_tree(self):
        return os.getenv('AETROS_GIT_WORK_DIR') or os.path.normpath(self.storage_dir + '/' + self.model_name + '/' + self.job_id)

    @property
    def env(self):
        my_env = os.environ.copy()
        if self.index_path:
            # todo, shouldn't it be the same as master's?
            my_env['GIT_INDEX_FILE'] = self.index_path

        my_env['GIT_SSH'] = os.getenv('GIT_SSH', '')

        return my_env

    @property
    def temp_path(self):
        return self.git_path + '/temp'

    @property
    def ref_head(self):
        return 'refs/aetros/job/' + self.job_id

    @property
    def git_url(self):
        return 'git@%s:%s.git' % (self.git_host, self.model_name)

    def get_base_command(self):
        base_command = ['git', '--bare', '--git-dir', self.git_path]
        base_command += ['-c', 'user.name=' + self.git_name]
        base_command += ['-c', 'user.email=' + self.git_email]

        return ' '.join(base_command)

    def command_exec(self, command, inputdata=None, allowed_to_fail=False, show_output=False, no_logging=False):
        interrupted = False

        if inputdata is not None and not isinstance(inputdata, six.binary_type):
            inputdata = inputdata.encode('utf-8')

        if command[0] != 'git':
            base_command = ['git', '--bare', '--git-dir', self.git_path]
            if command[0] == 'commit-tree' or command[0] == 'commit':
                base_command += ['-c', 'user.name=' + self.git_name]
                base_command += ['-c', 'user.email=' + self.git_email]
            command = base_command + command

        p = None
        stdoutdata = ''
        stderrdata = ''

        try:
            self.command_lock.acquire()

            if no_logging:
                p = subprocess.Popen(command, bufsize=0, stdin=subprocess.PIPE, env=self.env)
                p.communicate(inputdata)
            else:
                p = subprocess.Popen(
                    command, bufsize=0,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env
                )
                stdoutdata, stderrdata = p.communicate(inputdata)

                if show_output:
                    if stdoutdata:
                        sys.stdout.write(stdoutdata)
                    if stderrdata:
                        sys.stderr.write(stderrdata)
        except (KeyboardInterrupt, SystemExit):
            raise
        except TypeError as e:
            self.logger.error("Could not execute Git command: " + str(command))
            self.logger.error(str(self.env))
            self.logger.error(str(e))
            sys.exit(2)
        finally:
            if self.command_lock.locked():
                self.command_lock.release()

        try:
            stderrdata = stderrdata.decode('utf-8')
        except Exception: pass

        if os.getenv('DEBUG', 0) == '2':
            self.logger.debug("Git command: " + (' '.join(command)))

        # When working on Git in several threads, sometimes it can not get the lock file, like:
        #
        #   fatal: Unable to create '/Users/marc/.aetros/marcj/debug:test.git/ORIG_HEAD.lock': File exists.
        #
        #   Another git process seems to be running in this repository, e.g.
        #   an editor opened by 'git commit'. Please make sure all processes
        #   are terminated then try again. If it still fails, a git process
        #   may have crashed in this repository earlier:
        #   remove the file manually to continue.
        #
        # We neeed to check for that error, and run the command again

        if 'Another git process' in stderrdata:
            time.sleep(0.3)
            return self.command_exec(command, inputdata, allowed_to_fail)

        if 'Connection refused' in stderrdata or 'Permission denied' in stderrdata:
            if 'Permission denied' in stderrdata:
                self.logger.warning("You have no permission to push to that model. Make sure your SSH key is properly"
                                    " configured.")

            self.logger.error(stderrdata)
            return '', 1, ''

        if not interrupted and not allowed_to_fail and p is not None and p.returncode != 0:
            raise GitCommandException('Command failed: ' + ' '.join(command) + ', code: ' + str(p.returncode)
                                      +"\nstdout: '" + str(stdoutdata)
                                      +"',\nstderr: '" + str(stderrdata)
                                      # +"', env="+str(self.env)
                                      +", input="+str(inputdata)[:50])

        return stdoutdata, p.returncode if p is not None else None, stderrdata

    def prepare_index_file(self):
        """
        Makes sure that GIT index file we use per job (by modifying environment variable GIT_INDEX_FILE)
        is not locked and empty. Git.fetch_job uses `git read-tree` to updates this index. For new jobs, we start
        with an empty index - that's why we delete it every time.
        """
        if os.getenv('AETROS_GIT_INDEX_FILE'):
            self.index_path = os.getenv('AETROS_GIT_INDEX_FILE')
            return

        import tempfile
        h, path = tempfile.mkstemp('aetros-git', '', self.temp_path)

        self.index_path = path

        # we give git a unique file path for that index. However, git expect it to be non-existent for empty indexes.
        # empty file would lead to "fatal: index file smaller than expected"
        os.close(h)
        os.unlink(self.index_path)

        self.logger.debug('GIT_INDEX_FILE created at ' + self.index_path)

    def fetch_job(self, job_id, checkout=False):
        """
        Fetch the current job reference (refs/aetros/job/<id>) from origin and (when checkout=True)read its tree to
        the current git index and checkout into working director.
        """
        self.job_id = job_id

        self.logger.debug("Git fetch job reference %s" % (self.ref_head, ))
        out, code, err = self.command_exec(['ls-remote', 'origin', self.ref_head])

        if code:
            self.logger.error('Could not find the job ' + job_id + ' on the server. Are you online and does the job exist?')
            sys.exit(1)

        try:
            self.command_exec(['fetch', '-f', '-n', 'origin', self.ref_head+':'+self.ref_head])
        except Exception:
            self.logger.error("Could not load job information for " + job_id + '. You need to be online to start pre-configured jobs.')
            raise

        self.read_job(job_id, checkout)

    def is_job_fetched(self, job_id):
        try:
            self.command_exec(['rev-parse', 'refs/aetros/job/' + job_id])[0].decode('utf-8').strip()
            return True
        except Exception:
            return False

    def read_job(self, job_id, checkout=False):
        """
        Reads head and reads the tree into index,
        and checkout the work-tree when checkout=True.

        This does not fetch the job from the actual server. It needs to be in the local git already.
        """
        self.job_id = job_id

        commit = self.get_head_commit()
        self.logger.debug('Job ref points to ' + commit)
        self.command_exec(['read-tree', self.ref_head])

        if checkout:
            self.logger.debug('Working directory in ' + self.work_tree)

            # make sure we have checked out all files we have added until now. Important for simple models,
            # so we have the actual model.py and dataset scripts.
            if os.path.exists(self.work_tree):
                shutil.rmtree(self.work_tree)

            os.makedirs(self.work_tree)

            # make the working tree reflect exactly the tree of ref_head.
            # since we removed the dir before, we have exactly the tree of the reference
            self.command_exec(['--work-tree', self.work_tree, 'reset', '--hard', self.ref_head])

    def read_tree(self, ref):
        """
        Reads the ref into the current index.

        :param ref: the actual git reference
        :return:
        """
        self.command_exec(['read-tree', ref])

    # def restart_job(self):
    #     if not self.job_id:
    #         raise Exception('Could not restart unknown job. fetch_job() it first.')
    #
    #     self.command_exec(['update-ref', self.ref_head, self.job_id])
    #
    #     self.command_exec(['read-tree', self.ref_head])
    #
    #     # make sure we have checked out all files we have added until now. Important for simple models, so we have the
    #     # actual model.py and dataset scripts.
    #     if not os.path.exists(self.work_tree):
    #         os.makedirs(self.work_tree)
    #
    #     # updates index and working tree
    #     # this leaves other files in self.work_tree alone, which needs to be because this is also the working tree
    #     # of files checked out by start.py (custom models)
    #     self.command_exec(['--work-tree', self.work_tree, 'reset', '--hard', self.ref_head])

    # def create_task_id(self, job_id, data):
    #     """
    #     Creates a new task id and reference (refs/aetros/task/<id>) by creating a new commit with the same tree
    #     as job_id's and added aetros/task.json file. As parent commit the job_id is used.
    #     """
    #     self.read_tree(job_id)
    #     self.add_file('aetros/task.json', simplejson.dumps(data, indent=4))
    #     tree_id = self.write_tree()
    #
    #     task_id = self.command_exec(['commit-tree', '-m', "TASK_CREATED", tree_id, '-p', job_id])[0].decode('utf-8').strip()
    #
    #     ref = 'refs/aetros/task/' + task_id
    #     self.command_exec(['update-ref', ref, task_id])
    #
    #     return task_id

    def create_job_id(self, data):
        """
        Create a new job id and reference (refs/aetros/job/<id>) by creating a new commit with empty tree. That
        root commit is the actual job id. A reference is then created to the newest (head) commit of this commit history.
        The reference will always be updated once a new commit is added.
        """
        self.add_file('aetros/job.json', simplejson.dumps(data, indent=4))
        tree_id = self.write_tree()

        self.job_id = self.command_exec(['commit-tree', '-m', "JOB_CREATED", tree_id])[0].decode('utf-8').strip()

        out, code, err = self.command_exec(['show-ref', self.ref_head], allowed_to_fail=True)
        if not code:
            self.logger.warning("Generated job id already exists, because exact same experiment values given. Ref " + self.ref_head)

        self.command_exec(['update-ref', self.ref_head, self.job_id])

        # make sure we have checkedout all files we have added until now. Important for simple models, so we have the
        # actual model.py and dataset scripts.
        if not os.path.exists(self.work_tree):
            os.makedirs(self.work_tree)

        # updates index and working tree
        # this leaves other files in self.work_tree alone, which needs to be because this is also the working tree
        # of files checked out by start.py (custom models)
        self.command_exec(['--work-tree', self.work_tree, 'reset', '--hard', self.ref_head])

        # every caller needs to make sure to call git.push
        return self.job_id

    def start_push_sync(self):
        """
        Starts the detection of unsynced Git data.
        """
        self.active_thread = True
        self.active_push = True

        self.thread_push_instance = Thread(target=self.thread_push)
        self.thread_push_instance.daemon = True
        self.thread_push_instance.start()

    def stop(self):
        """
        Stops the `git push` thread and commits all streamed files (Git.store_file and Git.stream_file), followed
        by a final git push.
        
        You can not start the process again.
        """
        self.active_thread = False

        if self.thread_push_instance and self.thread_push_instance.isAlive():
            self.thread_push_instance.join()

        with self.batch_commit('STREAM_END'):
            for path, handle in six.iteritems(self.streamed_files.copy()):
                # open again and read full content
                full_path = os.path.normpath(self.temp_path + '/stream-blob/' + self.job_id + '/' + path)
                self.logger.debug('Git stream end for file: ' + full_path)

                del self.streamed_files[path]

                # make sure its written to the disk
                try:
                    self.stream_files_lock.acquire()
                    if not handle.closed:
                        handle.flush()
                        handle.close()
                finally:
                    self.stream_files_lock.release()

                with open(full_path, 'r') as f:
                    self.commit_file(path, path, f.read())

                if not self.keep_stream_files:
                    os.unlink(full_path)

        with self.batch_commit('STORE_END'):
            for path, bar in six.iteritems(self.store_files.copy()):
                full_path = os.path.normpath(self.temp_path + '/store-blob/' + self.job_id + '/' + path)
                self.logger.debug('Git store end for file: ' + full_path)

                del self.store_files[path]

                try:
                    self.stream_files_lock.acquire()
                    self.commit_file(path, path, open(full_path, 'r').read())
                finally:
                    self.stream_files_lock.release()

                if not self.keep_stream_files:
                    os.unlink(full_path)

    def clean_up(self):
        self.logger.debug("Git: clean up")

        if os.path.exists(self.index_path):
            os.remove(self.index_path)

        if self.delete_git_ssh:
            self.delete_git_ssh()
            self.delete_git_ssh = None

    def batch_commit(self, message):
        """
        Instead of committing a lot of small commits you can batch it together using this controller.

        Example:

        with git.batch_commit('BATCHED'):
            git.commit_file('my commit 1', 'path/to/file', 'content from file')
            git.commit_json_file('[1, 2, 3]', 'path/to/file2', 'json array') 

        Withing the `with` block you can use group the method calls of `commit_file` and `commit_json_file`, and every other
        method calling this two methods.

        :type message: str 
        :return: with controller to be used with Python's `with git.batch_commit():`
        """
        class controlled_execution:
            def __init__(self, git, message):
                self.git = git
                self.message = message

            def __enter__(self):
                self.git.git_batch_commit = True

                if self.git.job_id:
                    # make sure we're always on the tip tree
                    self.git.read_tree(self.git.ref_head)

            def __exit__(self, type, value, traceback):
                self.git.git_batch_commit = False

                # if nothing committed, we return early
                if not self.git.git_batch_commit_messages: return

                commit_message = self.message
                if self.git.git_batch_commit_messages:
                    commit_message = commit_message + "\n\n" + "\n".join(self.git.git_batch_commit_messages)
                self.git.git_batch_commit_messages = []

                self.git.commit_index(commit_message)

        return controlled_execution(self, message)

    def get_empty_tree_id(self):
        """
        Returns the famous empty tree id. To be used in creating a new empty root commit without any files.
        4b825dc642cb6eb9a060e54bf8d69288fbee4904

        :rtype: str 
        """
        return self.command_exec(['hash-object', '--stdin', '-ttree'], '')[0].decode('utf-8').strip()

    def store_file(self, path, data, fast_lane=True):
        """
        Store the file in temp folder and stream it to server if online. 
        
        This makes sure that we have all newest data of this file on the server directly. 
        
        This method always overwrites the content of path. If you want to append always the content, 
        use Git.stream_file() instead.
        
        At the end of the job, the content the server received is stored as git blob on the server. It is then committed 
        locally and pushed. Git detects that the server already has the version (through the continuous streaming)
        and won't push it again.
        """

        self.stream_files_lock.acquire()
        try:

            full_path = os.path.normpath(self.temp_path + '/store-blob/' + self.job_id + '/' + path)
            if not os.path.exists(os.path.dirname(full_path)):
                os.makedirs(os.path.dirname(full_path))

            if hasattr(data, 'encode'):
                data = data.encode("utf-8", 'replace')

            already_set = path in self.store_files and self.store_files[path] == data

            if is_debug2():
                sys.__stderr__.write('git:store_file(%s, %s, %s), already_set=%s\n'
                                     % (str(path), str(data)[0:180], str(fast_lane), str(already_set)))

            if already_set:
                return

            open(full_path, 'wb').write(data)
            self.store_files[path] = data

            if self.client.online is not False:
                self.client.send({'type': 'store-blob', 'path': path, 'data': data}, channel='' if fast_lane else 'files')
        finally:
            self.stream_files_lock.release()

    def stream_file(self, path, fast_lane=True):
        """
        Create a temp file, stream it to the server if online and append its content using the write() method. 
        This makes sure that we have all newest data of this file on the server directly.
        
        At the end of the job, the content the server received is stored as git blob on the server. It is then committed 
        locally and pushed. Git detects that the server already has the version (through the continuous streaming)
        and won't push it again. Very handy for rather large files that will append over time (like channel data, logs)
        
        Example:
        
        self.log_stream = git.stream_file('log.txt')
        
        self.log_stream.write("new line\n");
        self.log_stream.write("another line\n");
        """

        # create temp file
        # open temp file

        # register stream file and write locally
        # on end() git_commit that file locally

        # create socket connection to server
        # stream file to server
        # on end() send server end signal, so he can store its content in git as blob as well.
        # A git push would detect that both sides have the same content already,
        # except when server connection broke between start() and end().
        # Result -> already transmitted logs/channel data (probably many MBs) won't transfered twice
        # when doing a git-push.

        # return handler to write to this file

        full_path = os.path.normpath(self.temp_path + '/stream-blob/' + self.job_id + '/' + path)
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))

        handle = open(full_path, 'wb')
        self.streamed_files[path] = handle

        class Stream():
            def __init__(self, git):
                self.git = git

            def write(self, data):
                if path not in self.git.streamed_files:
                    # already committed to server
                    return

                if hasattr(data, 'encode'):
                    data = data.encode("utf-8", 'replace')

                try:
                    self.git.stream_files_lock.acquire()
                    if not handle.closed:
                        handle.write(data)
                        handle.flush()
                except IOError as e:
                    handle.close()

                    if 'No space left' in e.__str__():
                        sys.stderr.write(traceback.format_exc() + '\n')
                        self.git.logger.error(e.__str__())
                finally:
                    self.git.stream_files_lock.release()

                if self.git.client.online is not False:
                    self.git.client.send({'type': 'stream-blob', 'path': path, 'data': data}, channel='' if fast_lane else 'files')

        return Stream(self)

    def write_blob(self, content):
        return self.command_exec(['hash-object', '-w', "--stdin"], content)[0].decode('utf-8').strip()

    def add_index(self, mode, blob_id, path):
        """
        Add new entry to the current index
        :param tree: 
        :return: 
        """
        self.command_exec(['update-index', '--add', '--cacheinfo', mode, blob_id, path])

    def write_tree(self):
        """
        Writes the current index into a new tree
        :return: the tree sha
        """
        return self.command_exec(['write-tree'])[0].decode('utf-8').strip()

    def commit_json_file(self, message, path, content):
        return self.commit_file(message, path + '.json', simplejson.dumps(content, default=invalid_json_values))

    def lock_write(self):
        class Controller():
            def __init__(self, git):
                self.git = git

            def __enter__(self):
                self.git.index_lock.acquire()

                if self.git.job_id:
                    # make sure we're always on the tip tree
                    self.git.read_tree(self.git.ref_head)

            def __exit__(self, type, value, traceback):
                self.git.index_lock.release()

        return Controller(self)

    def add_file(self, path, content):
        """
        Add a new file as blob in the storage and add its tree entry into the index.
        
        :param path: str
        :param content: str
        """
        blob_id = self.write_blob(content)
        self.add_index('100644', blob_id, path)

    def add_file_path(self, path, work_tree, verbose=True):
        """
        Add a new file as blob in the storage and add its tree entry into the index.

        :param path: str
        :param content: str
        """
        args = ['--work-tree', work_tree, 'add', '-f']
        if verbose:
            args.append('--verbose')
        args.append(path)
        self.command_exec(args, show_output=verbose)

    def add_local_file(self, path):
        with open(path, 'r') as f:
            self.add_file(path, f.read())

    def commit_file(self, message, path, content):
        """
        Add a new file as blob in the storage, add its tree entry into the index and commit the index.

        :param message: str
        :param path: str
        :param content: str
        :return: 
        """
        if self.git_batch_commit:
            self.add_file(path, content)
            self.git_batch_commit_messages.append(message)
        else:
            with self.lock_write():
                if self.job_id:
                    self.read_tree(self.ref_head)

                self.add_file(path, content)

                return self.commit_index(message)

    def diff_objects(self, latest_commit_sha):
        """
                Push all changes to origin, based on objects, not on commits.
                Important: Call this push after every new commit, or we lose commits.
                """
        base = ['git', '--bare', '--git-dir', self.git_path]

        object_shas = []
        summary = {'commits': [], 'trees': [], 'files': []}

        def read_parents_and_tree_from(commit):
            if commit in self.synced_object_shas or commit in object_shas:
                # this commit has already been synced or read
                return None, None

            self.synced_object_shas[commit] = True
            summary['commits'].append(commit)
            object_shas.append(commit)

            object_content = subprocess.check_output(base + ['cat-file', '-p', commit]).decode('utf-8').strip()
            parents = []
            tree = ''
            for line in object_content.splitlines():
                if line.startswith('tree '):
                    tree = line[len('tree '):]
                if line.startswith('parent '):
                    parents.append(line[len('parent '):])

            return parents, tree

        def collect_files_from_tree(tree):
            if tree in self.synced_object_shas or tree in object_shas:
                # we have exactly this tree already synced or read, meaning all its objects as well
                return

            self.synced_object_shas[tree] = True
            summary['trees'].append(tree)
            object_shas.append(tree)

            object_content = subprocess.check_output(base + ['ls-tree', '-r', '-t', tree]).decode('utf-8').strip()

            for line in object_content.splitlines():
                exploded = line.split(' ')

                if len(exploded) < 3:
                    sys.stderr.write("Error: Wrong line format of ls-tree for %s: %s\n" % (tree, line,))
                    sys.exit(1)

                object_to_add = str(exploded[2][:40])
                path = str(exploded[2][41:])

                if object_to_add in self.synced_object_shas or object_to_add in object_shas:
                    # have it already in the list or already synced
                    continue

                object_shas.append(object_to_add)
                self.synced_object_shas[object_to_add] = True
                summary['files'].append([object_to_add, path])

        commits_to_check = [latest_commit_sha]

        while len(commits_to_check):
            sha = commits_to_check.pop(0)
            parents, tree = read_parents_and_tree_from(sha)

            if parents:
                for parent in parents:
                    if parent not in commits_to_check:
                        commits_to_check.append(parent)

            if tree:
                collect_files_from_tree(tree)

        self.logger.debug("shas_to_check %d: %s " % (len(object_shas), str(object_shas),))

        if not object_shas:
            return [], summary

        try:
            self.logger.debug("Do git-cat-file-check.sh")
            ssh_stream = create_ssh_stream(read_home_config(), exit_on_failure=False)
            channel = ssh_stream.get_transport().open_session()
            channel.exec_command('git-cat-file-check.sh "%s"' % (self.model_name + '.git',))
            channel.sendall('\n'.join(object_shas))
            channel.shutdown_write()

            def readall(c):
                content = b''
                while True:
                    try:
                        chunk = c.recv(1024)
                        if chunk == b'':
                            break
                        content += chunk
                    except (KeyboardInterrupt, SystemExit):
                        return

                return content

            missing_objects = readall(channel).decode('utf-8').splitlines()
            channel.close()
            ssh_stream.close()

            # make sure we have in summary only SHAs we actually will sync
            for stype in six.iterkeys(summary):
                ids = summary[stype][:]
                for sha in ids:
                    if stype == 'files':
                        if sha[0] not in missing_objects:
                            summary[stype].remove(sha)
                    else:
                        if sha not in missing_objects:
                            summary[stype].remove(sha)

            return missing_objects, summary
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            self.logger.error("Failed to generate diff_objects: %s" % (str(e),))
            for sha in object_shas:
                if sha in self.synced_object_shas:
                    del self.synced_object_shas[sha]
            return None, None

    def get_head_commit(self):
        return self.command_exec(['rev-parse', self.ref_head])[0].decode('utf-8').strip()

    def push(self):
        self.push_lock.acquire()
        missing_object_sha = []

        try:
            commit_sha = self.get_head_commit()
            missing_object_sha, summary = self.diff_objects(commit_sha)
            base = ['git', '--bare', '--git-dir', self.git_path]

            if not missing_object_sha:
                return False

            self.logger.debug("Found %d missing objects" % (len(missing_object_sha),))
            self.logger.debug(str(summary))

            if missing_object_sha:
                self.logger.debug("Git push")
                try:
                    opts = [
                        '--no-reuse-delta',
                        '--compression=0',
                    ]
                    pack_process = subprocess.Popen(base + ['pack-objects', '--stdout'] + opts, stderr=subprocess.PIPE,
                                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                    pack_content, stderr = pack_process.communicate(('\n'.join(missing_object_sha)).encode())

                    if pack_process.returncode:
                        self.logger.error("Git pack-objects failed: %s, error: %s" % (pack_content, stderr))
                        self.synced_object_shas = {}
                        return False

                    message = {
                        'type': 'git-unpack-objects',
                        'ref': self.ref_head,
                        'commit': commit_sha,
                        'pack': pack_content,
                        'objects': summary,
                    }
                    size = self.client.send(message, 'files')
                    self.logger.debug("Git pack of size %d is on the way" % (size,))
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    # next push we check again all objects in that job history
                    self.synced_object_shas = {}
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        finally:
            self.push_lock.release()

        return missing_object_sha

    def thread_push(self):
        last_synced_head = self.get_head_commit()

        while self.active_thread:
            try:
                head = self.get_head_commit()
                if last_synced_head != head:
                    self.logger.debug("Git head moved from %s to %s" % (last_synced_head, head))
                    if self.push() is not False:
                        last_synced_head = head

                time.sleep(0.5)
            except (SystemExit, KeyboardInterrupt):
                return
            except Exception as e:
                time.sleep(5)

    def commit_index(self, message):
        """
        Commit the current index.
        :param message: str
        :return: str the generated commit sha
        """
        tree_id = self.write_tree()

        args = ['commit-tree', tree_id, '-p', self.ref_head]

        # todo, this can end in a race-condition with other processes adding commits
        commit = self.command_exec(args, message)[0].decode('utf-8').strip()
        self.command_exec(['update-ref', self.ref_head, commit])

        return commit

    def has_file(self, path):
        try:
            out, code, err = self.command_exec(['cat-file', '-p', self.ref_head+':'+path])

            return code == 0
        except Exception:
            return False

    def contents(self, path):
        """
        Reads the given path of current ref_head and returns its content as utf-8
        """
        try:
            out, code, err = self.command_exec(['cat-file', '-p', self.ref_head+':'+path])
            if not code:
                return out.decode('utf-8')
        except Exception:
            pass

        return None

    def git_read(self, path):
        return self.command_exec(['cat-file', '-p', self.ref_head+':'+path])
