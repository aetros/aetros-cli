import os
from threading import Thread, Lock

import requests

import signal
import json
import time

from io import BytesIO
from requests.auth import HTTPBasicAuth


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


class EventListener:
    def __init__(self):
        self.events = {}

    def on(self, name, callback):
        if name not in self.events:
            self.events[name] = []

        self.events[name].append(callback)

    def fire(self, name):
        if name in self.events:
            for callback in self.events[name]:
                callback()


class AetrosBackend:
    def __init__(self, job_id = None):
        self.event_listener = EventListener()
        self.api_token = os.getenv('API_KEY')

        self.job_id = job_id
        self.queue = []
        self.queueLock = Lock()
        self.active_syncer = True

        self.host = os.getenv('API_HOST')
        if not self.host or self.host == 'false':
            self.host = 'aetros.com'

        self.thread = Thread(target=self.syncer)
        self.thread.daemon = True
        self.thread.start()
        self.in_request = False
        self.stop_requested = False

    def syncer(self):
        while (self.active_syncer):
            self.sync()
            time.sleep(1)

    def sync(self):
        self.queueLock.acquire()

        if len(self.queue) == 0:
            self.queueLock.release()
            return

        max_messages_per_request = 20

        queue_to_save = self.queue[:max_messages_per_request]
        self.queue = self.queue[max_messages_per_request:]
        self.queueLock.release()

        # print "Sync %d/%d items to server" % (len(queue_to_save), len(self.queue))
        self.in_request = True
        try:
            response = self.post('job/sync', json={'id': self.job_id, 'items': queue_to_save})
            failed = False
            if response.status_code != 200:
                failed = True
            else:
                if response.content:
                    content = json.loads(response.content)
                    if 'not_exists' in content:
                        print "Job deleted meanwhile. Stop current job."
                        os.kill(os.getpid(), signal.SIGINT)
                    elif 'stop' in content:
                        if not self.stop_requested:
                            print "Job stopped through trainer."
                            self.stop_requested = True
                            os.kill(os.getpid(), signal.SIGINT)
                    else:
                        if 'result' in content and content['result'] is True:
                            if len(self.queue) > 0:
                                # we have still some messages, so send directly next sync
                                self.sync()
                        else:
                            failed = True
                else:
                    failed = True

            if failed:
                self.queueLock.acquire()
                self.queue = queue_to_save + self.queue
                self.queueLock.release()
                print "Error in sending job information: %s" % (response.content,)

        except Exception:
            self.queueLock.acquire()
            self.queue = queue_to_save + self.queue
            self.queueLock.release()

        self.in_request = False

    def stop_syncer(self):
        self.active_syncer = False

        while self.in_request:
            time.sleep(0.05)

        self.queueLock.acquire()
        queue = self.queue[:]
        self.queueLock.release()

        if len(queue) > 0:
            print "Sending last (%d) monitoring information to server ... " % (len(queue),)
            response = self.post('job/sync', json={'id': self.job_id, 'items': queue})
            if response.status_code != 200:
                print "Error in sending job information."

    def kill(self):
        self.active_syncer = False

    def get_url(self, affix):

        url = 'http://%s/api/%s' % (self.host, affix)

        if self.api_token:
            if '?' in url:
                url += '&token=' + self.api_token
            else:
                url += '?token=' + self.api_token

        return url

    def write_log(self, id, message):
        item = {'id': id, 'message': message}

        self.queueLock.acquire()
        self.queue.append({
            'type': 'log',
            'time': time.time(),
            'data': item
        })
        self.queueLock.release()

    def upload_weights(self, name, file_path, accuracy=None, with_status=False):
        if not os.path.isfile(file_path):
            return

        files = {}
        files[name] = open(file_path, 'r')

        class CancelledError(Exception):
            def __init__(self, msg):
                self.msg = msg
                Exception.__init__(self, msg)

            def __str__(self):
                return self.msg

            __repr__ = __str__

        class BufferReader(BytesIO):
            def __init__(self, buf=b'',
                         callback=None,
                         cb_args=(),
                         cb_kwargs={}):
                self._callback = callback
                self._cb_args = cb_args
                self._cb_kwargs = cb_kwargs
                self._progress = 0
                self._len = len(buf)
                BytesIO.__init__(self, buf)

            def __len__(self):
                return self._len

            def read(self, n=-1):
                chunk = BytesIO.read(self, n)
                self._progress += int(len(chunk))
                self._cb_kwargs.update({
                    'size': self._len,
                    'progress': self._progress
                })
                if self._callback:
                    try:
                        self._callback(*self._cb_args, **self._cb_kwargs)
                    except:
                        raise CancelledError('The upload was cancelled.')
                return chunk

        state = {'progress': -1}

        def progress(size=None, progress=None):
            current_progress = int(progress / size * 100)
            if state['progress'] != current_progress:
                state['progress'] = current_progress

                if with_status:
                    self.set_status('UPLOAD WEIGHTS ' + str(current_progress) + '%')
                else:
                    print("{0}%".format(current_progress))

        files = {"upfile": (name, open(file_path, 'rb').read())}

        (data, ctype) = requests.packages.urllib3.filepost.encode_multipart_formdata(files)

        headers = {
            "Content-Type": ctype
        }

        body = BufferReader(data, progress)

        url = self.get_url('job/weights?id=%s&accuracy=%f.2' % (self.job_id, accuracy if accuracy is not None else -1))
        response = requests.post(url, data=body, headers=headers)

        if response.status_code != 200:
            raise Exception('Uploading of weights failed: %d: %s' % (response.status_code,response.content))

    def set_status(self, status):
        print 'Training status changed to %s ' % (status,)
        self.job_add_status('status', status)

    def get_best_weight_url(self, job_id):
        response = self.get('job/weight-best', {'id': job_id})
        return response.json()

    def get(self, url, params=None, **kwargs):
        return requests.get(self.get_url(url), params=params, **kwargs)

    def post(self, url, data=None, **kwargs):
        return requests.post(self.get_url(url), data=data, **kwargs)

    def put(self, url, data=None, **kwargs):
        return requests.put(self.get_url(url), data=data, **kwargs)

    def create_job(self, name, server_id='local', dataset_id=None, insights=False):
        response = self.put('job',
                            {'networkId': name, 'serverId': server_id, 'insights': insights, 'datasetId': dataset_id})
        if response.status_code != 200:
            raise Exception("Could not create training: %s" % (response.content,))

        job_id = response.json()
        self.job_id = job_id

        return job_id

    def ensure_network(self, network_name, model_json, settings=None, network_type='custom', layers=None, graph=None):
        response = self.put('network/ensure', {
            'id': network_name,
            'type': network_type,
            'model': model_json,
            'settings': json.dumps(settings, allow_nan=False) if settings else None,
            'layers': json.dumps(layers, allow_nan=False),
            'graph': json.dumps(graph, allow_nan=False),
        })

        if response.status_code != 200:
            raise Exception("Could not create network: %s" % (response.content,))

        return True

    def get_job(self):
        response = self.get('job', {'id': self.job_id})
        if response.status_code != 200:
            raise Exception("Could not find version: %s" % (response.content,))

        job = response.json()

        if  job is None or 'error' in job:
            raise Exception('Training/Version not found. Have you configured your token correctly? %s: %s' % (job['error'], job['message']))

        return response.json()

    def get_light_job(self):
        response = self.get('job', {'id': self.job_id, 'light': 1})
        if response.status_code != 200:
            raise Exception("Could not find version (%s): %s" % (self.job_id, response.content,))

        job = response.json()

        if job is None or job == 'Job not found':
            raise Exception('Version not found. Have you configured your token correctly?')

        if 'error' in job:
            raise Exception('Version not found. Have you configured your token correctly? %s: %s' % (job['error'], job['message']))

        if not isinstance(job, dict):
            raise Exception('Version does not exist. Make sure you created the job via AETROS TRAINER')

        if not len(job['config']):
            raise Exception('Version does not have a configuration. Make sure you created the job via AETROS TRAINER')

        return job

    def job_started(self, id, pid):
        return self.post('job/started', {'id': id, 'pid': pid})

    def job_add_status(self, statusKey, statusValue):
        item = {'statusKey': statusKey, 'statusValue': json.dumps(statusValue, allow_nan=False)}

        self.queueLock.acquire()
        self.queue.append({
            'type': 'job-status',
            'time': time.time(),
            'data': item
        })
        self.queueLock.release()

    def job_set_info_key(self, key, info):
        self.queueLock.acquire()
        self.queue.append({
            'type': 'job-info',
            'time': time.time(),
            'data': {'key': key, 'info': info}
        })
        self.queueLock.release()

    def job_add_insight(self, info, images):
        self.queueLock.acquire()
        self.queue.append({
            'type': 'job-insight',
            'time': time.time(),
            'data': {'info': info, 'images': images}
        })
        self.queueLock.release()
