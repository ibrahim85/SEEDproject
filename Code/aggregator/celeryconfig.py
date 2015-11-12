
from datetime import timedelta
from celery.schedules import crontab

CELERYBEAT_SCHEDULE = {
 'aggregate monthly': {
  'task': 'tasks.aggregate',
  'schedule': crontab(0, 0, day_of_month=1)
 },
}
