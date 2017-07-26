import json
import os
import subprocess
import six
from threading import Thread, Lock

import time

from aetros.utils import invalid_json_values

class GitCommandException(Exception):
    pass

class Git:
    def __init__(self, logger, client, git_host, git_path, model_name):
        self.logger = logger
        self.client = client
        self.git_host = git_host
        self.git_path = git_path
        self.model_name = model_name

        self.debug = False

        self.job_id = None
        self.online = True

        self.git_batch_push = False
        self.git_batch_commit = False

        self.git_batch_commit_messages = []
        self.git_last_commit = None

        self.streamed_files = {}
        self.store_files = {}

        if os.path.exists(self.git_path):
            # check if its a git repo
            p = subprocess.Popen(
                ['git', '--bare', '--git-dir', self.git_path, 'remote', 'get-url', 'origin'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            p.wait()
            if p.returncode != 0:
                raise Exception('Given git_path (%s) already exists and does not seem to be a git repository. Error: %s' % (self.git_path, p.stderr.read()))
        else:
            self.command_exec(['git', 'clone', '--bare', self.git_url, self.git_path])

        # check if given repo_path is current folder.
        # check its origin remote and see if model_name matches
        origin_url = self.command_exec(['git', '--bare', '--git-dir', self.git_path, 'remote', 'get-url', 'origin'], allowed_to_fail=True).strip().decode("utf-8")
        if origin_url and ':' + model_name + '.git' not in origin_url:
            raise Exception("Given git_path (%s) points to a repository (%s) that is not the git repo of the model (%s). " % (self.git_path, origin_url, model_name))

        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        self.logger.info("Started tracking of files in git %s for remote %s" % (self.git_path, origin_url))

        self.active_thread = True
        self.thread_push = Thread(target=self.thread_push)
        self.thread_push.daemon = True
        self.thread_push.start()

    @property
    def env(self):
        my_env = os.environ.copy()
        if self.job_id:
            my_env['GIT_INDEX_FILE'] = self.index_path

        return my_env

    @property
    def index_path(self):
        return self.temp_path + '/git_index_job_' + self.job_id

    def thread_push(self):
        while self.active_thread:
            if self.job_id and self.online:
                self.command_exec(['git', '--bare', '--git-dir', self.git_path, 'push', '-f', 'origin', self.ref_head])

            time.sleep(1)

    @property
    def temp_path(self):
        return self.git_path + '/temp'

    @property
    def ref_head(self):
        return 'refs/aetros/job/' + self.job_id + '/head'

    @property
    def git_url(self):
        return 'git@%s:%s.git' % (self.git_host, self.model_name)

    def command_exec(self, command, inputdata=None, allowed_to_fail=False):
        self.logger.debug("[Debug] Git command: " + (' '.join(command)) + ', input=' + str(inputdata))

        p = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env
        )

        stdoutdata, stderrdata = p.communicate(inputdata)

        self.logger.debug("[Debug] Git command stdout: " + str(stdoutdata) + ', stderr: '+ str(stderrdata))

        if not allowed_to_fail and p.returncode != 0:
            raise GitCommandException('Command failed: ' + ' '.join(command) + ', code: ' + str(p.returncode)+"\nstderr: '" + str(stderrdata)+"', input="+str(inputdata))

        return stdoutdata

    def prepare_index_file(self):
        if os.path.exists(self.index_path + '.lock'):
            os.remove(self.index_path + '.lock')

        if os.path.exists(self.index_path):
            os.remove(self.index_path)


    def fetch_job(self, job_id):
        self.job_id = job_id

        self.prepare_index_file()

        self.logger.info("Git fetch job reference %s" % (self.ref_head, ))
        self.command_exec(['git', '--bare', '--git-dir', self.git_path, 'fetch', '-f', '-n', 'origin', self.ref_head+':'+self.ref_head])
        self.command_exec(['git', '--bare', '--git-dir', self.git_path, 'read-tree', self.ref_head])

        with open(self.git_path + '/' + self.ref_head, 'r') as f:
            self.git_last_commit = f.read().strip()

    def create_job_id(self):

        self.job_id = self.command_exec(['git', '--bare', '--git-dir', self.git_path, 'commit-tree', '-m', "JOB_CREATED", self.get_empty_tree_id()]).strip().decode("utf-8")
        self.git_last_commit = self.job_id
        self.prepare_index_file()

        self.command_exec(['git', '--bare', '--git-dir', self.git_path, 'update-ref', self.ref_head, self.git_last_commit])

        self.logger.info("Job git ref created " + self.ref_head)

        self.push()

        return self.job_id

    def stop(self):
        self.active_thread = False

        with self.batch_commit('STREAM_END'):
            for path, handle in six.iteritems(self.streamed_files):
                full_path = self.temp_path + '/stream-blob/' + self.job_id + '/' + path
                handle.seek(0)
                self.commit_file(path, path, handle.read())
                handle.close()
                os.unlink(full_path)
            self.streamed_files = {}

        with self.batch_commit('STORE_END'):
            for path, bar in six.iteritems(self.store_files):
                full_path = self.temp_path + '/store-blob/' + self.job_id + '/' + path
                self.commit_file(path, path, open(full_path, 'r').read())
                os.unlink(full_path)
            self.store_files = {}

        self.push()

        if os.path.exists(self.index_path):
            os.remove(self.index_path)

    def batch_commit(self, message):
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

    def batch_push(self):
        class controlled_execution:
            def __init__(self, job):
                self.job = job

            def __enter__(self):
                self.job.git_batch_push = True

            def __exit__(self, type, value, traceback):
                self.job.git_batch_push = False
                self.job.push()

        return controlled_execution(self)

    def get_empty_tree_id(self):
        return self.command_exec(['git', '--bare', '--git-dir', self.git_path, 'hash-object', '--stdin', '-ttree'], '').strip().decode('utf-8')

    def store_file(self, path, data):
        """
        Stores the file in temp folder and uploads to server if online. At the end of the job, it is committed in Git.
        :param path: 
        :param content: 
        :return: 
        """

        full_path = self.temp_path + '/store-blob/' + self.job_id + '/' + path
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))

        open(full_path, 'w+').write(data)
        self.store_files[path] = True

        self.client.send({'type': 'store-blob', 'path': path, 'data': data})

    def stream_file(self, path):
        """
        Creates a temp stream and streams to the server if online. At the end of the job, it is committed in Git.
        :param path: 
        :return: 
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
            def __init__(self, handle, client):
                self.handle = handle
                self.client = client

            def write(self, data):
                try:
                    handle.write(data)
                    self.client.send({'type': 'stream-blob', 'path': path, 'data': data})
                except:
                    pass

        return Stream(handle, self.client)

    def write_blob(self, content):
        return self.command_exec(['git', '--bare', '--git-dir', self.git_path, 'hash-object', '-w', "--stdin"], content).decode('utf-8').strip()

    def add_index(self, tree):
        """
        Add new entry to the current index
        :param tree: 
        :return: 
        """
        self.command_exec(['git', '--bare', '--git-dir', self.git_path, 'update-index', '--add', '--cacheinfo', tree])

    def write_tree(self):
        """
        Writes the current index into a new tree
        :return: the tree sha
        """
        return self.command_exec(['git', '--bare', '--git-dir', self.git_path, 'write-tree']).decode("utf-8") .strip()

    def commit_json_file(self, message, path, content):
        return self.commit_file(message, path + '.json', json.dumps(content, default=invalid_json_values))

    def add_file(self, path, content):
        """
        Add a new file as blob in the storage and add its tree entry into the index.
        
        :param path: str
        :param content: str
        """
        blob_id = self.write_blob(content)
        tree = '100644,' + blob_id + ',' + path
        self.add_index(tree)

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
        if self.git_batch_push:
            return

        self.command_exec(['git', '--bare', '--git-dir', self.git_path, 'push', 'origin', self.ref_head])

    def commit_index(self, message):
        """
        Commit the current index.
        :param message: str
        :return: str the generated commit sha
        """
        tree_id = self.write_tree()

        self.git_last_commit = self.command_exec([
            'git',
            '--bare',
            '--git-dir',
            self.git_path,
            'commit-tree',
            tree_id,
            '-p',
            self.git_last_commit
        ], message).decode("utf-8") .strip()

        # update ref
        self.command_exec(['git', '--bare', '--git-dir', self.git_path, 'update-ref', self.ref_head, self.git_last_commit])

        return self.git_last_commit

    def git_job_last_commit_sha(self):
        if not self.job_id:
            raise Exception('Can not fetch last commit sha, when no job_id is set.')

        output = self.command_exec(['git', '--bare', '--git-dir', self.git_path, 'show-ref', self.ref_head]).decode('utf-8').strip()
        if output:
            return output.split(' ')[0]
        else:
            raise Exception('Job ref not created yet. ' + self.ref_head)

    def git_read(self, path):
        return self.command_exec(['git', '--bare', '--git-dir', self.git_path, 'cat-file', '-p', self.ref_head+':'+path])
