import os
from threading import Thread, Lock

import requests

import signal
import json
import time

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
    def __init__(self, job_id):
        self.event_listener = EventListener()
        self.api_token = os.getenv('API_KEY')
        self.job_id = job_id
        self.queue = []
        self.queueLock = Lock()
        self.active_syncer = True

        self.host = os.getenv('API_HOST') if os.getenv('API_HOST') else 'aetros.com'

        self.thread = Thread(target=self.syncer)
        self.thread.daemon = True
        self.thread.start()
        self.in_request = False
        self.stop_requested = False

        self.auth = HTTPBasicAuth('stage', 'moep')

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

    def upload_weights(self, name, file_path):
        if not os.path.isfile(file_path):
            return

        files = {}
        files[name] = open(file_path, 'r')
        return requests.post(self.get_url('job/weights?id=%s' % (self.job_id,)), auth=self.auth, files=files)

    def get_best_weight_url(self, job_id):
        response = self.get('job/weight-best', {'id': job_id})
        return response.json()

    def get(self, url, params=None, **kwargs):
        return requests.get(self.get_url(url), params=params, auth=self.auth, **kwargs)

    def post(self, url, data=None, **kwargs):
        return requests.post(self.get_url(url), data=data, auth=self.auth, **kwargs)

    def put(self, url, data=None, **kwargs):
        return requests.put(self.get_url(url), data=data, auth=self.auth, **kwargs)

    def create_job(self, name, server_id ='local', dataset_id = None, insights = False):
        response =  self.put('job', {'networkId': name, 'serverId': server_id, 'insights': insights, 'datasetId': dataset_id})
        if response.status_code != 200:
            print("Could not create training: %s" % (response.content, ))
            return None

        return response.json()

    def get_job(self):
        response = self.get('job', {'id': self.job_id})
        if response.status_code != 200:
            print("Could not find training: %s" % (response.content,))
            return None

        return response.json()

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
