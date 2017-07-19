import json
import os
import subprocess

import six

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

        self.git_batch_commit = False
        self.git_batch_commit_messages = []
        self.git_last_commit = None

        self.streamed_files = []

    @property
    def git_url(self):
        return 'git@%s:%s' % (self.git_host, self.model_name)

    def command_exec(self, command, inputdata=None):
        self.logger.debug("[Debug] Git command: " + (' '.join(command)) + ', input=' + str(inputdata))

        p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        stdoutdata, stderrdata = p.communicate(inputdata)

        self.logger.debug("[Debug] Git command stdout: " + str(stdoutdata) + ', stderr: '+ str(stderrdata))

        # if p.returncode is None:
        #     p.wait()

        if p.returncode != 0:
            raise Exception('Command failed: ' + ' '.join(command) + ', code: ' + str(p.returncode)+"\n" + str(stderrdata))

        return stdoutdata


    def create_job_id(self):

        if not os.path.exists(self.git_path):
            self.command_exec(['git', 'clone', self.git_url, self.git_path])

        self.job_id = self.command_exec(['git', '-C', self.git_path, 'commit-tree', '-m', "JOB_START", self.get_empty_tree_id()]).strip().decode("utf-8")
        self.git_last_commit = self.job_id

        os.environ['GIT_INDEX_FILE'] = '.temp/git_index_job_' + self.job_id
        index_path = self.git_path + '/' + os.environ['GIT_INDEX_FILE']
        if os.path.exists(index_path + '.lock'):
            os.remove(index_path + '.lock')

        self.command_exec(['git', '-C', self.git_path, 'update-ref', 'refs/aetros/job/' + self.job_id, self.git_last_commit])

        self.logger.info("Job git ref created " + 'refs/aetros/job/' + self.job_id)

        return self.job_id

    def stop(self):

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
                commit_message = self.message
                if self.job.git_batch_commit_messages:
                    commit_message = commit_message + "\n\n" + "\n".join(self.job.git_batch_commit_messages)
                self.job.git_batch_commit_messages = []

                self.job.commit_index(commit_message)

        return controlled_execution(self, message)

    def get_empty_tree_id(self):
        return self.command_exec(['git', '-C', self.git_path, 'hash-object', '--stdin', '-ttree'], '').strip().decode('utf-8')

    def git_stream_file(self, path):

        # create temp file
        # open temp file

        # register stream file and write locally
        # on end() git_commit that file

        # create socket connection to server
        # stream file to server
        # on end() send server end signal, so he can store its content in git as blob as well. A git push would detect
        # that both sides have the same content already, except when server connection broke between start() and end().

        # return handler to write to this file

        full_path = self.git_path + '/.temp/stream-blobs/' + self.job_id + '/' + path
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))

        self.streamed_files.append(full_path)

        handle = open(full_path, 'w+')

        def write(data):
            handle.write(data)
            self.client.send({'type': 'stream-blob', 'path': path, 'data': data})

        return write

    def write_blob(self, content):
        return self.command_exec(['git', '-C', self.git_path, 'hash-object', '-w', "--stdin"], content).decode('utf-8').strip()

    def add_file(self, tree):
        self.command_exec(['git', '-C', self.git_path, 'update-index', '--add', '--cacheinfo', tree])

    def write_tree(self):
        return self.command_exec(['git', '-C', self.git_path, 'write-tree']).decode("utf-8") .strip()

    def commit_file(self, message, path, content):
        if not isinstance(content, six.string_types):
            content = json.dumps(content, default=invalid_json_values)

        blob_id = self.write_blob(content)
        tree = '100644,' + blob_id + ',' + path
        self.add_file(tree)

        if not self.git_batch_commit:
            self.commit_index(message)
        else:
            self.git_batch_commit_messages.append(message)

    def commit_index(self, message):
        tree_id = self.write_tree()

        self.git_last_commit = self.command_exec(['git', '-C', self.git_path, 'commit-tree', tree_id, '-p', self.git_last_commit], message).decode("utf-8") .strip()

        # update ref
        self.command_exec(['git', '-C', self.git_path, 'update-ref', 'refs/aetros/job/' + self.job_id, self.git_last_commit])

        return self.git_last_commit

    def git_job_last_commit_sha(self):
        if not self.job_id:
            raise Exception('Can not fetch last commit sha, when no job_id is set.')

        output = self.command_exec(['git', '-C', self.git_path, 'show-ref', 'refs/aetros/job/' + self.job_id]).decode('utf-8').strip()
        if output:
            return output.split(' ')[0]
        else:
            raise Exception('Job ref not created yet. ' + 'refs/aetros/job/' + self.job_id)

    def git_read(self, path):
        last_commit = self.git_job_last_commit_sha()

        output = self.command_exec(['git', '-C', self.git_path, 'ls-tree', last_commit, path]).decode("utf-8").strip()
        if output:
            blob_id = output.split(' ')[2].split('\t')[0]

            return self.command_exec(['git', '-C', self.git_path, 'cat-file', blob_id])
