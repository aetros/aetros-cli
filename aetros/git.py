import json
import os
import subprocess
import six
from threading import Thread, Lock

import time

from aetros.utils import invalid_json_values

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

        if not os.path.exists(git_path):
            os.makedirs(git_path)

        if not os.path.exists(git_path + '/.temp'):
            os.makedirs(git_path + '/.temp')

        self.streamed_files = {}
        self.store_files = {}

        self.active_thread = True
        self.thread_push = Thread(target=self.thread_push)
        self.thread_push.daemon = True
        self.thread_push.start()

    def thread_push(self):
        while self.active_thread:
            if self.job_id and self.online:
                self.command_exec(['git', '-C', self.git_path, 'push', 'origin', self.ref_head])

            time.sleep(1)

    @property
    def ref_head(self):
        return 'refs/aetros/job/' + self.job_id + '/head'

    @property
    def git_url(self):
        return 'git@%s:%s' % (self.git_host, self.model_name)

    def command_exec(self, command, inputdata=None):
        self.logger.debug("[Debug] Git command: " + (' '.join(command)) + ', input=' + str(inputdata))

        p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        stdoutdata, stderrdata = p.communicate(inputdata)

        self.logger.debug("[Debug] Git command stdout: " + str(stdoutdata) + ', stderr: '+ str(stderrdata))

        if p.returncode != 0:
            raise Exception('Command failed: ' + ' '.join(command) + ', code: ' + str(p.returncode)+"\nstderr: '" + str(stderrdata)+"', input="+str(inputdata))

        return stdoutdata

    def create_job_id(self):

        if not os.path.exists(self.git_path + '/.git'):
            self.command_exec(['git', 'clone', self.git_url, self.git_path])

        self.job_id = self.command_exec(['git', '-C', self.git_path, 'commit-tree', '-m', "JOB_CREATED", self.get_empty_tree_id()]).strip().decode("utf-8")
        self.git_last_commit = self.job_id

        os.environ['GIT_INDEX_FILE'] = '.temp/git_index_job_' + self.job_id
        index_path = self.git_path + '/' + os.environ['GIT_INDEX_FILE']
        if os.path.exists(index_path + '.lock'):
            os.remove(index_path + '.lock')

        self.command_exec(['git', '-C', self.git_path, 'update-ref', self.ref_head, self.git_last_commit])

        self.logger.info("Job git ref created " + self.ref_head)

        return self.job_id

    def stop(self):
        self.active_thread = False

        with self.batch_commit('STREAM_END'):
            for path, handle in six.iteritems(self.streamed_files):
                full_path = self.git_path + '/.temp/stream-blob/' + self.job_id + '/' + path
                handle.seek(0)
                self.commit_file(path, path, handle.read())
                handle.close()
                os.unlink(full_path)
            self.streamed_files = {}

        with self.batch_commit('STORE_END'):
            for path, bar in six.iteritems(self.store_files):
                full_path = self.git_path + '/.temp/store-blob/' + self.job_id + '/' + path
                self.commit_file(path, path, open(full_path, 'r').read())
                os.unlink(full_path)
            self.store_files = {}

        self.push()

        if 'GIT_INDEX_FILE' in os.environ:
            index_path = self.git_path + '/' + os.environ['GIT_INDEX_FILE']
            if os.path.exists(index_path):
                os.remove(index_path)

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
        return self.command_exec(['git', '-C', self.git_path, 'hash-object', '--stdin', '-ttree'], '').strip().decode('utf-8')

    def store_file(self, path, data):
        """
        Stores the file in temp folder and uploads to server if online. At the end of the job, it is committed in Git.
        :param path: 
        :param content: 
        :return: 
        """

        full_path = self.git_path + '/.temp/store-blob/' + self.job_id + '/' + path
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

        full_path = self.git_path + '/.temp/stream-blob/' + self.job_id + '/' + path
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))

        handle = open(full_path, 'w+')
        self.streamed_files[path] = handle

        class Stream():
            def __init__(self, handle, client):
                self.handle = handle
                self.client = client

            def write(self, data):
                handle.write(data)
                self.client.send({'type': 'stream-blob', 'path': path, 'data': data})

        return Stream(handle, self.client)

    def write_blob(self, content):
        return self.command_exec(['git', '-C', self.git_path, 'hash-object', '-w', "--stdin"], content).decode('utf-8').strip()

    def add_file(self, tree):
        self.command_exec(['git', '-C', self.git_path, 'update-index', '--add', '--cacheinfo', tree])

    def write_tree(self):
        return self.command_exec(['git', '-C', self.git_path, 'write-tree']).decode("utf-8") .strip()

    def commit_json_file(self, message, path, content):
        return self.commit_file(message, path, json.dumps(content, default=invalid_json_values))

    def commit_file(self, message, path, content):

        blob_id = self.write_blob(content)
        tree = '100644,' + blob_id + ',' + path
        self.add_file(tree)

        if not self.git_batch_commit:
            self.commit_index(message)
        else:
            self.git_batch_commit_messages.append(message)

    def push(self):
        if self.git_batch_push:
            return

        self.command_exec(['git', '-C', self.git_path, 'push', 'origin', self.ref_head])

    def commit_index(self, message):
        tree_id = self.write_tree()

        self.git_last_commit = self.command_exec(['git', '-C', self.git_path, 'commit-tree', tree_id, '-p', self.git_last_commit], message).decode("utf-8") .strip()

        # update ref
        self.command_exec(['git', '-C', self.git_path, 'update-ref', self.ref_head, self.git_last_commit])

        return self.git_last_commit

    def git_job_last_commit_sha(self):
        if not self.job_id:
            raise Exception('Can not fetch last commit sha, when no job_id is set.')

        output = self.command_exec(['git', '-C', self.git_path, 'show-ref', self.ref_head]).decode('utf-8').strip()
        if output:
            return output.split(' ')[0]
        else:
            raise Exception('Job ref not created yet. ' + self.ref_head)

    def git_read(self, path):
        last_commit = self.git_job_last_commit_sha()

        output = self.command_exec(['git', '-C', self.git_path, 'ls-tree', last_commit, path]).decode("utf-8").strip()
        if output:
            blob_id = output.split(' ')[2].split('\t')[0]

            return self.command_exec(['git', '-C', self.git_path, 'cat-file', blob_id])
