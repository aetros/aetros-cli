__version__ = '0.9.5'
__prog__ = "aetros"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class JOB_STATUS:
    PROGRESS_STATUS_CREATED = 0
    PROGRESS_STATUS_QUEUED = 1
    PROGRESS_STATUS_STARTED = 2
    PROGRESS_STATUS_ENDED = 3
    PROGRESS_STATUS_ABORTED = 4
    PROGRESS_STATUS_CRASHED = 5