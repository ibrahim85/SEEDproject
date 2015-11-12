# Configuration file for the scheduler

from celery.schedules import crontab
from datetime import timedelta

CELERY_IMPORTS=("tasks",)
CELERYD_CONCURRENCY=1
CELERYBEAT_SCHEDULE = {
  'test': {
  'task': 'tasks.doParser',       # scheduled task, the scheduler will call doParser() function periodically
  'schedule': timedelta(days=1),  # interval between two scheduled calls, 1 day
  'args': (),                     # doParser() don't need arguments
 },
}

