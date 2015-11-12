from celery import Celery
from celery import task 
import datetime
import requests
import json
import sys
import psycopg2

#configure celery
celery = Celery('tasks', broker='amqp://guest@localhost//')
celery.config_from_object('celeryconfig')

#aggregate monthly data kairos and push it to postgres. data represents the aggregate query
def aggr_sum_metric(data):
    headers = {'content-type': 'application/json'}
    url = 'http://localhost:8013/api/v1/datapoints/query'
    data = json.dumps(data)
	#run query
    r = requests.post(url, data=data, headers=headers)
	#length of output array. Should be 1 per group since it's monthly aggregation and we are querying only for one month
    print len(r.json()['queries'][0]['results'])
    length =  len(r.json()['queries'][0]['results'])
	
	#Retrieve required values from array and call posgres insert function
    for num in range(0, length):
		gb_bldg_snapshot_id =  r.json()['queries'][0]['results'][num]['tags']['building_snapshot_id'][0]
        gb_mtr_id =  r.json()['queries'][0]['results'][num]['tags']['usage_point_id'][0]
        gb_energy_type_id =  r.json()['queries'][0]['results'][num]['tags']['usage_kind'][0]
        gb_timestamp =  r.json()['queries'][0]['results'][num]['values'][0][0]
        gb_agg_reading = r.json()['queries'][0]['results'][num]['values'][0][1]
		#push data to postgres
		insert_into_postgres(gb_bldg_snapshot_id,gb_mtr_id,gb_energy_type_id,gb_timestamp,gb_agg_reading,tsMonthStart,tsMonthEnd)
    else:
	print "End of for loop"
    
#insert into postgres	
def insert_into_postgres(gb_bldg_snapshot_id,gb_mtr_id,gb_energy_type_id,gb_timestamp,gb_agg_reading,tsMonthStart,tsMonthEnd):
	try: 
		conn = psycopg2.connect("dbname='seed-deploy' user='seed-admin' host='localhost' password='SEEDDB@architecture.cmu.edu'")
	except psycopg2.Error as e:
    		print "Unable to connect to database"
    		print e.pgerror
    		print e.diag.message_detail
    		sys.exit(1)
	cur = conn.cursor()
	#retrieve meter_id from seed_meter using buildingsnapshot_id, green_button_meter_id, energy_type
	cur.execute('SELECT seed_meter.id FROM seed_meter,seed_meter_building_snapshot WHERE seed_meter_building_snapshot.meter_id=seed_meter.id AND seed_meter_building_snapshot.buildingsnapshot_id=%(param_bs)s AND seed_meter.gb_meter_id=%(param_mtr_id)s AND seed_meter.energy_type=%(param_enrgy_type)s',{'param_bs': gb_bldg_snapshot_id,'param_enrgy_type': gb_energy_type_id,'param_mtr_id': gb_mtr_id})
	rows = cur.fetchall()
	
	#insert in seed_timeseries
	for row in rows:
		gb_timestamp_endtime = gb_timestamp
		print row[0]
    	cur.execute(INSERT INTO seed_timeseries (begin_time, end_time, reading, cost, meter_id) VALUES(%s, %s, %s, %s, %s),(tsMonthStart, tsMonthEnd, gb_agg_reading, '', row[0]))
	else:
		print "Loop ended"

@task
def aggregate():
    print "Starting Aggregation" 
	#find out last month's start and end timestamps
    monthlist = [1,3,5,7,8,10,12]
    today = datetime.datetime.today()
	#last month
    if today.month == 1:
      lastmonth = today.replace(year=(today.year - 1))
      lastmonth = lastmonth.replace(month=12)
    else:
      lastmonth = today.replace(month = (today.month - 1))
	
    #first day of last month
    firstDayOfLastMonth = lastmonth.replace(day=1).replace(hour=0).replace(minute=0).replace(second=0).replace(microsecond=0)
	
	#last day of the month
    if lastmonth.month in monthlist:
		lastDayOfLastMonth = lastmonth.replace(day=31)
    elif (lastmonth.month == 2) and (lastmonth.year%4 !=0):
		lastDayOfLastMonth = lastmonth.replace(day=28)
    elif (lastmonth.month == 2) and (lastmonth.year%4 ==0):
        lastDayOfLastMonth = lastmonth.replace(day=29)
    else:
		lastDayOfLastMonth = lastmonth.replace(day=30)
    lastDayOfLastMonth = lastDayOfLastMonth.replace(hour=23).replace(minute=59).replace(second=59).replace(microsecond=999999)
    
	#timestamps
	tsMonthStart = int(firstDayOfLastMonth.strftime("%s")) * 1000
    tsMonthEnd = int(lastDayOfLastMonth.strftime("%s")) * 1000
    
    #kairos aggregation query
    agg_query = {
                 "start_absolute": str(tsMonthStart),
                 "end_absolute": str(tsMonthEnd),

                 "metrics":[
                   {
                    "name": "gb_test",
					"aggregators": [
                        {
                            "name": "sum",
                            "sampling": {
                               "value": 1,
                               "unit": "months"
                            }
                        }
                     ],
                    "group_by": [
                       {
                           "name": "tag",
                           "tags": ["usage_kind","building_snapshot_id","usage_point_id","interval"]
                       }
                    ]
                    }
                  ]
               }
	#aggregate data using the agg_query
    aggr_sum_metric(agg_query)