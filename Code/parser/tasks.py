from celery import Celery
from celery import task

import psycopg2
import time
import sys
from datetime import date, timedelta

import GreenButtonDriver as driver
import GreenButtonDataAnalyser as analyser
import KairosInsert as tsdb

celery = Celery('tasks', broker='amqp://guest@localhost//')
celery.config_from_object('celeryconfig')

# Start celerybeat
# CLI command: celery -A tasks worker --loglevel=info --beat
# 

# convert database query result into an object
class reg(object):
    def __init__(self, cursor_des, registro):
        for (attr, val) in zip((d[0] for d in cursor_des), registro):
            setattr(self, attr, val)

#Format of date_str is MM/DD/YYYY
def increment_day(date_str):
    if date_str == '' or date_str == None:
        newdate = date.today()-timedelta(1)
    else: 
        t=time.strptime(date_str, '%m/%d/%Y')
        newdate=date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(1)
    
    return newdate.strftime('%m/%d/%Y')


@task
def doParser():
    try: 
        conn = psycopg2.connect("dbname='seed-deploy' user='seed-admin' host='127.0.0.1' password='SEEDDB@architecture.cmu.edu'")
    except psycopg2.Error as e:
        print "Unable to connect to database"
        print e.pgerror
        print e.diag.message_detail
        return

    cur = conn.cursor()
	
	# retrieve GreenButton URL and building_id relationship from PostgreSQL
    cur.execute('select * from ts_parser_record where active=\'Y\'')
    
    rows = cur.fetchall()
    cur_des = cur.description   # get all the table columns
    for row in rows:
        r = reg(cur_des, row)
		
		# retrieve one relationship data
        row_id = r.id
        url = r.url
        last_date = r.last_date
        last_ts = r.last_ts
        min_date_parameter = r.min_date_parameter
        max_date_parameter = r.max_date_parameter
        building_id = r.building_id
		
        incr_day =  increment_day(last_date)

		# assemble GreenButton URL
        url = url+min_date_parameter+"="+incr_day+"&"+max_date_parameter+"="+incr_day

        print 'Fetching url '+url

		# get parsed GreenButton data
        ts_data = driver.get_gb_data(url, building_id)
       
        print 'data fetched'
 
        if ts_data!=None:
			# pass parsed GreenButton data to analyser
            ts_data = analyser.data_analyse(ts_data)
			
            print 'inserting ts data'
			
			# insert data has interval less than monthly into Kairos database
            tsdb.insert(ts_data)
			
            print 'update db record: update ts_parser_record set last_date=\''+incr_day+'\' where id='+str(row_id)
			
            #update db record
            cur.execute('update ts_parser_record set last_date=\''+incr_day+'\' where id='+str(row_id))
            conn.commit()
			
            print 'update db finished'
        else:
            print '----ts_data is empty'
    
    
    cur.close()
    conn.close()
    print 'connection closed'
