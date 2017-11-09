import json
import os
import subprocess

import six
from threading import Thread, Lock
import time
import sys
from ruamel import yaml

from aetros.api import ApiConnectionError
from aetros.utils import invalid_json_values, setup_git_ssh


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
    def __init__(self, logger, client, config, model_name):
        self.logger = logger
        self.client = client

        self.config = config
        self.git_host = config['host']
        self.storage_dir = config['storage_dir']

        self.model_name = model_name

        self.git_path = os.path.normpath(self.storage_dir + '/' + model_name + '.git')

        self.command_lock = Lock()
        self.stream_files_lock = Lock()
        self.debug = False
        self.last_push_time = 0
        self.active_push = False
        self.index_path = None

        # dirty means, the git repository has changed and need a push
        self.dirty = False

        self.job_id = None
        self.online = True
        self.active_thread = False
        self.thread_push_instance = None

        self.git_batch_commit = False

        self.git_batch_commit_messages = []
        self.git_last_commit = None

        self.keep_stream_files = False

        self.streamed_files = {}
        self.store_files = {}

        self.prepare_index_file()

        if subprocess.Popen(['git', '--version'], stdout=subprocess.PIPE).wait() > 0:
            raise Exception("Git binary not available. Please install Git v2 first.")

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

        # check if given repo_path is current folder.
        # check its origin remote and see if model_name matches
        self.origin_url = self.get_remote_url('origin')

        if self.origin_url and self.git_url not in self.origin_url:
            raise Exception("Given git_path (%s) points to a repository (%s) that is not the git repo of the model (%s). "
                            "It seems you switched between aetros.com and an on-premise installation or update aetros.yml:host."
                            "Please remove the Git repository." % (self.git_path, self.origin_url, self.git_url))

        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

    def prepare_git_user(self):
        """
        Tries to read the git name and email, so all git commits have correct author.
        Requests /api/user-git to check which user is behind the current configured ssh key.
        """
        import aetros.api
        try:
            response = aetros.api.request('user-git')
            if response:
                user = yaml.safe_load(response)

                self.git_name = user['name']
                self.git_email = user['email']
            else:
                self.go_offline()
        except ApiConnectionError as e:
            self.go_offline()


    def get_remote_url(self, origin_name):
        output = self.command_exec(['remote', '-v'], allowed_to_fail=True)[0].decode('utf-8').strip()

        import re
        match = re.match('^' + re.escape(origin_name) + '\t([^\s]+)', output)
        if match:
            return match.group(1)

    @property
    def work_tree(self):
        return os.getenv('AETROS_GIT_WORK_DIR') or os.path.normpath(self.storage_dir + '/' + self.model_name + '/' + self.job_id)

    @property
    def env(self):
        my_env = os.environ.copy()
        if self.index_path:
            my_env['GIT_INDEX_FILE'] = self.index_path

        my_env['GIT_SSH'] = os.getenv('GIT_SSH')

        return my_env

    def thread_push(self):
        while self.active_thread:
            try:
                time.sleep(1)

                if self.job_id and self.online and self.active_push and self.dirty:
                    self.dirty = False
                    start = time.time()
                    self.command_exec(['push', '-f', 'origin', self.ref_head])
                    self.last_push_time = time.time() - start

            except SystemExit:
                return
            except KeyboardInterrupt:
                return

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

        return ''.join(base_command)

    def command_exec(self, command, inputdata=None, allowed_to_fail=False):
        interrupted = False

        if isinstance(inputdata, six.string_types):
            inputdata = six.b(inputdata)

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

            p = subprocess.Popen(
                command, bufsize=0,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env
            )

            stdoutdata, stderrdata = p.communicate(inputdata)
        except KeyboardInterrupt:
            raise
        finally:
            self.command_lock.release()

        try:
            stderrdata = stderrdata.decode('utf-8')
        except Exception: pass

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

            self.go_offline()
            self.logger.error(stderrdata)
            return '', 1, ''

        if not interrupted and not allowed_to_fail and p is not None and p.returncode != 0:
            raise GitCommandException('Command failed: ' + ' '.join(command) + ', code: ' + str(p.returncode)
                                      +"\nstdout: '" + str(stdoutdata)
                                      +"',\nstderr: '" + str(stderrdata)
                                      # +"', env="+str(self.env)
                                      +", input="+str(inputdata))

        return stdoutdata, p.returncode if p is not None else None, stderrdata

    def go_offline(self):
        """
        Go offline means disable all online communication and just store the data in local git.
        """
        if self.client:
            self.client.go_offline()

        self.online = False

    def prepare_index_file(self):
        """
        Makes sure that GIT index file we use per job (by modifying environment variable GIT_INDEX_FILE)
        is not locked and empty. Git.fetch_job uses `git read-tree` to updates this index. For new jobs, we start
        with an empty index - that's why we delete it every time.
        """
        import tempfile
        h, path = tempfile.mkstemp('aetros-git')

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
        Reads head and sets self.git_last_commit, reads the tree into index,
        and checkout the work-tree when checkout=True.

        This does not fetch the job from the actual server. It needs to be in the local git already.
        """
        self.job_id = job_id

        self.git_last_commit = self.command_exec(['rev-parse', self.ref_head])[0].decode('utf-8').strip()
        self.logger.debug('Job ref points to ' + self.git_last_commit)
        self.command_exec(['read-tree', self.ref_head])

        if checkout:
            self.logger.debug('Working directory in ' + self.work_tree)

            # make sure we have checked out all files we have added until now. Important for simple models,
            # so we have the actual model.py and dataset scripts.
            if not os.path.exists(self.work_tree):
                os.makedirs(self.work_tree)

            # updates index and working tree
            # this leaves other files in self.work_tree alone, which is necessary because this is also the working tree
            # of files checked out by start.py (custom models)
            self.command_exec(['--work-tree', self.work_tree, 'reset', '--hard', self.ref_head])

    def read_tree(self, ref):
        """
        Reads the ref into the current index and points last commit_id to its head.

        :param ref: the actual git reference
        :return:
        """
        self.command_exec(['read-tree', ref])
        self.git_last_commit = self.command_exec(['rev-parse', ref])[0].decode('utf-8').strip()

    def restart_job(self):
        if not self.job_id:
            raise Exception('Could not restart unknown job. fetch_job() it first.')

        self.command_exec(['update-ref', self.ref_head, self.job_id])
        self.dirty = True

        self.command_exec(['read-tree', self.ref_head])

        # make sure we have checked out all files we have added until now. Important for simple models, so we have the
        # actual model.py and dataset scripts.
        if not os.path.exists(self.work_tree):
            os.makedirs(self.work_tree)

        # updates index and working tree
        # this leaves other files in self.work_tree alone, which needs to be because this is also the working tree
        # of files checked out by start.py (custom models)
        self.command_exec(['--work-tree', self.work_tree, 'reset', '--hard', self.ref_head])

    def create_job_id(self, data):
        """
        Create a new job id and reference (refs/aetros/job/<id>) by creating a new commit with empty tree. That
        root commit is the actual job id. A reference is then created to the newest (head) commit of this commit history.
        The reference will always be updated once a new commit is added.
        """
        self.add_file('aetros/job.json', json.dumps(data, indent=4))
        tree_id = self.write_tree()

        self.job_id = self.command_exec(['commit-tree', '-m', "JOB_CREATED", tree_id])[0].decode('utf-8').strip()
        self.git_last_commit = self.job_id

        out, code, err = self.command_exec(['show-ref', self.ref_head], allowed_to_fail=True)
        if not code:
            self.logger.warning("Generated job id already exists, because exact same experiment values given. Ref " + self.ref_head)

        self.command_exec(['update-ref', self.ref_head, self.git_last_commit])

        # make sure we have checkedout all files we have added until now. Important for simple models, so we have the
        # actual model.py and dataset scripts.
        if not os.path.exists(self.work_tree):
            os.makedirs(self.work_tree)

        # updates index and working tree
        # this leaves other files in self.work_tree alone, which needs to be because this is also the working tree
        # of files checked out by start.py (custom models)
        self.command_exec(['--work-tree', self.work_tree, 'reset', '--hard', self.ref_head])
        self.dirty = True

        return self.job_id

    def start(self):
        """
        Start the git push thread
        """
        if self.active_thread:
            return

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
            if self.thread_push_instance and self.thread_push_instance.isAlive():
                self.thread_push_instance.join()

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

        :rtype: str 
        """
        return self.command_exec(['hash-object', '--stdin', '-ttree'], '')[0].decode('utf-8').strip()

    def store_file(self, path, data):
        """
        Store the file in temp folder and stream it to server if online. 
        
        This makes sure that we have all newest data of this file on the server directly. 
        
        This method always overwrites the content of path. If you want to append always the content, 
        use Git.stream_file() instead.
        
        At the end of the job, the content the server received is stored as git blob on the server. It is then committed 
        locally and pushed. Git detects that the server already has the version (through the continuous streaming)
        and won't push it again.
        
        :param path: 
        :param content: 
        :return: 
        """
        try:
            self.stream_files_lock.acquire()

            full_path = os.path.normpath(self.temp_path + '/store-blob/' + self.job_id + '/' + path)
            if not os.path.exists(os.path.dirname(full_path)):
                os.makedirs(os.path.dirname(full_path))

            open(full_path, 'w+').write(data)
            self.store_files[path] = True

            if self.online:
                self.client.send({'type': 'store-blob', 'path': path, 'data': data})
        finally:
            self.stream_files_lock.release()

    def stream_file(self, path):
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
        
        :param path: 
        :rtype: Stream class
        :return Returns a instance with a `write(data)` method.
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

        handle = open(full_path, 'w+')
        self.streamed_files[path] = handle

        class Stream():
            def __init__(self, git):
                self.git = git

            def write(self, data):
                if path not in self.git.streamed_files:
                    # already committed to server
                    return

                try:
                    self.git.stream_files_lock.acquire()
                    if not handle.closed:
                        handle.write(data)
                        handle.flush()
                finally:
                    self.git.stream_files_lock.release()


                if self.git.online:
                    self.git.client.send({'type': 'stream-blob', 'path': path, 'data': data})

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
        return self.commit_file(message, path + '.json', json.dumps(content, default=invalid_json_values))

    def add_file(self, path, content):
        """
        Add a new file as blob in the storage and add its tree entry into the index.
        
        :param path: str
        :param content: str
        """
        blob_id = self.write_blob(content)
        self.add_index('100644', blob_id, path)

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
        if not self.git_batch_commit:
            if self.job_id:
                self.read_tree(self.ref_head)

            self.add_file(path, content)

            return self.commit_index(message)
        else:
            self.add_file(path, content)
            self.git_batch_commit_messages.append(message)

    def push(self):
        """
        Push all changes to origin
        """
        try:
            self.command_exec(['push', 'origin', '-f', self.ref_head])
            return True
        except Exception as e:
            # this may fail due to:
            #     stderr: 'remote: error: cannot lock ref 'refs/aetros/job/cc3114813659d443c8e4f9682517067a1e9ec9ff':
            #     is at 31785d1c126b24cabd1948cba2e126912393b8e6 but expected 3f7186391884bd097c09c31567ef49718fc271a5
            #
            self.logger.warning(str(e))
            return False

    def commit_index(self, message):
        """
        Commit the current index.
        :param message: str
        :return: str the generated commit sha
        """
        tree_id = self.write_tree()

        args = ['commit-tree', tree_id, '-p', self.ref_head]

        # todo, this can end in a race-condition with other processes adding commits
        self.git_last_commit = self.command_exec(args, message)[0].decode('utf-8').strip()
        self.command_exec(['update-ref', self.ref_head, self.git_last_commit])
        self.dirty = True

        return self.git_last_commit

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
