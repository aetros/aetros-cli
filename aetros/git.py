import json
import os
import subprocess

import signal
import six
from threading import Thread, Lock

import time

import sys

from aetros.api import ApiConnectionError
from aetros.utils import invalid_json_values


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
    def __init__(self, logger, client, git_host, storage_dir, model_name):
        self.logger = logger
        self.client = client
        self.git_host = git_host
        self.storage_dir = storage_dir
        self.model_name = model_name

        self.git_path = self.storage_dir + '/' + model_name + '.git'

        self.debug = False
        self.last_push_time = 0
        self.active_push = False
        self.index_path = None

        self.job_id = None
        self.online = True
        self.active_thread = False
        self.thread_push_instance = None

        self.git_batch_commit = False

        self.git_batch_commit_messages = []
        self.git_last_commit = None

        self.streamed_files = {}
        self.store_files = {}

        self.prepare_index_file()

        self.logger.debug("GIT_SSH='" + str(os.getenv('GIT_SSH'))+"'")
        self.git_name = None
        self.git_email = None

        import aetros.api
        try:
            user = json.loads(aetros.api.request('user-git'))

            self.git_name = user['name']
            self.git_email = user['email']
        except ApiConnectionError as e:
            self.online = False

        if not self.git_name:
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
        origin_url = self.get_remote_url('origin')

        if origin_url and ':' + self.model_name + '.git' not in origin_url:
            raise Exception("Given git_path (%s) points to a repository (%s) that is not the git repo of the model (%s). " % (self.git_path, origin_url, model_name))

        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        self.logger.info("Started tracking of job files in git %s for remote %s" % (self.git_path, origin_url))

    def get_remote_url(self, origin_name):
        output = self.command_exec(['remote', '-v'], allowed_to_fail=True)[0].decode('utf-8').strip()

        import re
        match = re.match('^' + re.escape(origin_name) + '\t([^\s]+)', output)
        if match:
            return match.group(1)

    @property
    def work_tree(self):
        return self.storage_dir + '/' + self.model_name + '/' + self.job_id

    @property
    def env(self):
        my_env = os.environ.copy()
        if self.index_path:
            my_env['GIT_INDEX_FILE'] = self.index_path

        return my_env

    def thread_push(self):
        while self.active_thread:
            try:
                if self.job_id and self.online and self.active_push:
                    start = time.time()
                    self.command_exec(['push', '-f', 'origin', self.ref_head])
                    self.last_push_time = time.time() - start

                time.sleep(1)
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

    def command_exec(self, command, inputdata=None, allowed_to_fail=False):
        interrupted = False

        if isinstance(inputdata, six.string_types):
            inputdata = six.b(inputdata)

        if command[0] != 'git':
            base_command = ['git', '--bare', '--git-dir', self.git_path]
            base_command += ['-c', 'user.name=' + self.git_name]
            base_command += ['-c', 'user.email=' + self.git_email]
            command = base_command + command

        p = None
        stdoutdata = ''
        stderrdata = ''

        def preexec_fn():
            # Ignore the SIGINT signal by setting the handler to the standard
            # signal handler SIG_IGN.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            p = subprocess.Popen(
                command, preexec_fn=preexec_fn, bufsize=0,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env
            )

            stdoutdata, stderrdata = p.communicate(inputdata)
        except KeyboardInterrupt:
            try:
                if p is not None:
                    stdoutdata = p.stdout.read()
                    stderrdata = p.stderr.read()
            except:
                pass
            interrupted = True

        input = ''

        try:
            input = inputdata.decode('utf-8')
        except:
            input = 'binary'

        try:
            stderrdata = stderrdata.decode('utf-8')
        except: pass

        message = "Git command: " + (' '.join(command))
        # message += "\nstdout: " + str(stdoutdata.decode('utf-8')) + ', stderr: '+ str(stderrdata)

        self.logger.debug(message)

        if 'Connection refused' in stderrdata or 'Permission denied' in stderrdata:
            if 'Permission denied' in stderrdata:
                self.logger.warning("You have no permission to push to that model.")

            self.online = False
            self.logger.error(stderrdata)
            return '', 1, ''

        if not interrupted and not allowed_to_fail and p is not None and p.returncode != 0:
            raise GitCommandException('Command failed: ' + ' '.join(command) + ', code: ' + str(p.returncode)+"\nstdout: '" + str(stdoutdata)+"',\nstderr: '" + str(stderrdata)+"', input="+str(inputdata))

        return stdoutdata, p.returncode, stderrdata

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
        os.unlink(self.index_path)

        self.logger.debug('GIT_INDEX_FILE created at ' + self.index_path)

    def fetch_job(self, job_id):
        """
        Fetch the current job reference (refs/aetros/job/<id>) from origin and read its tree to the current git index.
        
        :type job_id: str 
        """
        self.job_id = job_id

        self.logger.info("Git fetch job reference %s" % (self.ref_head, ))
        out, code, err = self.command_exec(['ls-remote', 'origin', self.ref_head])

        if code:
            self.logger.error('Could not find the job ' + job_id + ' on the server. Are you online and does the job exist?')
            sys.exit(1)

        try:
            self.command_exec(['fetch', '-f', '-n', 'origin', self.ref_head+':'+self.ref_head])
        except:
            self.logger.error("Could not load job information for " + job_id + '. You need to be online to start pre-configured jobs.')
            raise

        self.git_last_commit = job_id
        self.command_exec(['update-ref', self.ref_head, self.git_last_commit])
        self.command_exec(['read-tree', self.ref_head])

        # make sure we have checked out all files we have added until now. Important for simple models, so we have the
        # actual model.py and dataset scripts.
        if not os.path.exists(self.work_tree):
            os.makedirs(self.work_tree)

        self.logger.info('Working directory in ' + self.work_tree)
        self.command_exec(['--work-tree', self.work_tree, 'checkout', self.ref_head, '.'])

    def create_job_id(self, data):
        """
        Create a new job id and reference (refs/aetros/job/<id>) by creating a new commit with empty tree. That
        root commit is the actual job id. A reference is then created to the newest (head) commit of this commit history.
        The reference will always be updated once a new commit is added.
        """
        self.add_file('aetros/job.json', json.dumps(data))
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

        self.command_exec(['--work-tree', self.work_tree, 'checkout', self.ref_head])

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
        
        This removes also the current git index file. You can not start the process again.
        """
        self.active_thread = False

        with self.batch_commit('STREAM_END'):
            for path, handle in six.iteritems(self.streamed_files.copy()):
                full_path = self.temp_path + '/stream-blob/' + self.job_id + '/' + path
                handle.seek(0)
                self.commit_file(path, path, handle.read())
                handle.close()
                os.unlink(full_path)
            self.streamed_files = {}

        with self.batch_commit('STORE_END'):
            for path, bar in six.iteritems(self.store_files.copy()):
                full_path = self.temp_path + '/store-blob/' + self.job_id + '/' + path
                self.commit_file(path, path, open(full_path, 'r').read())
                os.unlink(full_path)
            self.store_files = {}

    def clean_up(self):
        if os.path.exists(self.index_path):
            os.remove(self.index_path)

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
            def __init__(self, job, message):
                self.job = job
                self.message = message

            def __enter__(self):
                self.job.git_batch_commit = True

            def __exit__(self, type, value, traceback):
                self.job.git_batch_commit = False

                # if nothing committed, we return early
                if not self.job.git_batch_commit_messages: return

                commit_message = self.message
                if self.job.git_batch_commit_messages:
                    commit_message = commit_message + "\n\n" + "\n".join(self.job.git_batch_commit_messages)
                self.job.git_batch_commit_messages = []

                self.job.commit_index(commit_message)

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

        full_path = self.temp_path + '/store-blob/' + self.job_id + '/' + path
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))

        open(full_path, 'w+').write(data)
        self.store_files[path] = True

        if self.online:
            self.client.send({'type': 'store-blob', 'path': path, 'data': data})

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
        # on end() git_commit that file

        # create socket connection to server
        # stream file to server
        # on end() send server end signal, so he can store its content in git as blob as well. A git push would detect
        # that both sides have the same content already, except when server connection broke between start() and end().

        # return handler to write to this file

        full_path = self.temp_path + '/stream-blob/' + self.job_id + '/' + path
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))

        handle = open(full_path, 'w+')
        self.streamed_files[path] = handle

        class Stream():
            def __init__(self, handle, git):
                self.handle = handle
                self.git = git

            def write(self, data):
                try:
                    handle.write(data)
                    if self.git.online:
                        self.git.client.send({'type': 'stream-blob', 'path': path, 'data': data})
                except:
                    pass

        return Stream(handle, self)

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

    def commit_file(self, message, path, content):
        """
        Add a new file as blob in the storage, add its tree entry into the index and commit the index.
         
        :param message: str 
        :param path: str
        :param content: str
        :return: 
        """
        self.add_file(path, content)

        if not self.git_batch_commit:
            self.commit_index(message)
        else:
            self.git_batch_commit_messages.append(message)

    def push(self):
        """
        Push all changes to origin
        """
        try:
            self.command_exec(['push', 'origin', '-f', self.ref_head])
            return True
        except:
            return False

    def commit_index(self, message):
        """
        Commit the current index.
        :param message: str
        :return: str the generated commit sha
        """
        tree_id = self.write_tree()

        args = ['commit-tree', tree_id]

        if self.git_last_commit:
            # if not define, it creates a new root commit
            args += ['-p', self.git_last_commit]

        self.git_last_commit = self.command_exec(args, message)[0].decode('utf-8').strip()

        # update ref
        self.command_exec(['update-ref', self.ref_head, self.git_last_commit])

        return self.git_last_commit

    def git_job_last_commit_sha(self):
        if not self.job_id:
            raise Exception('Can not fetch last commit sha, when no job_id is set.')

        out, code = self.command_exec(['show-ref', self.ref_head])
        if not code:
            return out.decode('utf-8').strip().split(' ')[0]
        else:
            raise Exception('Job ref not created yet. ' + self.ref_head)

    def git_read(self, path):
        return self.command_exec(['cat-file', '-p', self.ref_head+':'+path])
