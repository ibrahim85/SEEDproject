import datetime
import time
import requests
import json
from json import JSONEncoder
import sys
import psycopg2
import pytz
from pytz import timezone
from geopy.geocoders import Nominatim
from geopy.distance import vincenty

url = "https://128.2.109.159/piwebapi/dataservers/s0-MYhSMORGkyGTe9bdohw0AV0lOLTYyTlBVMkJWTDIw/points?namefilter=*underground/"
filePath = "/home/seed-admin/Shilpi/degreedays/weather_station_mapping.csv"

def getStationId(city, state):
	with open(filePath) as inf:
		for line in inf:
			line_words = line.split(',')
			if(line_words[0] == city and (line_words[2] == state.upper() or line_words[3] == state.upper())):
				return line_words[1]
		else:
			distance = 9999999
			station = ''
			geolocator = Nominatim()
			location = geolocator.geocode(city+','+state)
			if location is not None:
				with open(filePath) as inf:
					for newline in inf:
						newline_words = newline.split(',')
						temp_distance = (vincenty((location.latitude, location.longitude), (newline_words[4], newline_words[5])).miles)
						if(temp_distance < distance):
							station = newline_words[1]
							distance = temp_distance
							if(distance < 11):
								print station + ':' + newline_words[0]
								return station
								break
						else:
							return 'NA'
			else:
				return 'NA'

def getUpperRange(lowerRange):
	month = lowerRange.month
	day = lowerRange.day
	year = lowerRange.year
        if (month in [1,3,5,7,8,10,12]):
                dayDelta = 31 - day + 1
        elif (month in [4,6,9,11]):
                dayDelta = 30 - day + 1
        elif (month == 2 and year%4 == 0):
                dayDelta = 29 - day + 1
        else:
                dayDelta = 28 - day + 1
	upperRange = lowerRange + datetime.timedelta(days=dayDelta,seconds=-1)
        return upperRange

def getTotalTemp(rec):
	tempCount = 0
	tempAggregate = 0
	numberOfRecords = len(rec.json()['Items'])
	for num in range(0, numberOfRecords):
                        tempCount = tempCount + 1
                        tempAggregate = tempAggregate + rec.json()['Items'][num]['Value']
	else:
		return tempAggregate/tempCount

def getCDDHDD(minDate,maxDate,city,state,baseTempCdd,baseTempHdd):
        print 'in getCDDHDD'
        stationId = getStationId(city,state)
        if(stationId != 'NA'):
                url =  "https://128.2.109.159/piwebapi/dataservers/s0-MYhSMORGkyGTe9bdohw0AV0lOLTYyTlBVMkJWTDIw/points?namefilter=*underground/*"+stationId+"*tempe*"
                r = requests.get(url, auth=('Weather', 'Weather1!@'), verify=False)
                webId = r.json()['Items'][0]['WebId']
                lowerRange = datetime.datetime.strptime(minDate,'%d %b %Y %H:%M:%S')
                upperRange = getUpperRange(lowerRange)
                tempMonthlyDict = {lowerRange.strftime('%b-%Y'):'0:0'}
                while lowerRange < datetime.datetime.strptime(maxDate,'%d %b %Y %H:%M:%S'):
                        lowerRangeString = lowerRange.strftime('%d %b %Y %H:%M:%S')
                        if upperRange < datetime.datetime.strptime(maxDate,'%d %b %Y %H:%M:%S'):
                                upperRangeString = upperRange.strftime('%d %b %Y %H:%M:%S')
                        else:
                                upperRangeString = maxDate
                        recordUrl = "https://128.2.109.159/piwebapi/streams/"+webId+"/recorded?starttime='"+lowerRangeString+"'&endtime='"+upperRangeString+"'&maxcount=149000"
                        rec = requests.get(recordUrl, auth=('Weather', 'Weather1!@'), verify=False)
                        avgDailyTempString = getTotalTemp(rec)
                        tempMonthlyDict[lowerRange.strftime('%b-%Y')] = avgDailyTempString
                        lowerRange = upperRange +  datetime.timedelta(seconds=1)
                        upperRange = getUpperRange(lowerRange)
                for key in tempMonthlyDict:
                        avgMonthlyTemperature = float(tempMonthlyDict.get(key))
                        cdd = max(0,(avgMonthlyTemperature - float(baseTempCdd)))
                        hdd = max(0,(float(baseTempHdd) - avgMonthlyTemperature))
                        tempMonthlyDict[key] = str(round(avgMonthlyTemperature,2))+':'+str(round(cdd,2))+':'+str(round(hdd,2))
                return tempMonthlyDict
        else:
                return 'NA'


def getBuildingDetails(host,pgresdbname,pgresuser,pgrespwd,buildingsnapshotid):
        conn_string = "host='"+str(host)+"' dbname='"+str(pgresdbname)+"' user='"+str(pgresuser)+"' password='"+str(pgrespwd)+"'"
        try:
                conn = psycopg2.connect(conn_string)
        except psycopg2.Error as e:
                print "Unable to connect to database"
                print e.pgerror
                print e.diag.message_detail
                sys.exit(1)
        cur = conn.cursor()

        #retrieve energy details from seed_timeseries for building using buildingsnapshot_id
        cur.execute('SELECT city,state_province,gross_floor_area FROM seed_buildingsnapshot WHERE seed_buildingsnapshot.id = %(param_bs)s',{'param_bs': buildingsnapshotid})

        rows = cur.fetchall()
        bldgDetailsArray = []
        #insert in seed_timeseries
        for row in rows:
                bldgDetailsArray.append(row[0])
                bldgDetailsArray.append(row[1])
		bldgDetailsArray.append(row[2])
        else:
                return bldgDetailsArray


def getMinMaxDates(host,pgresdbname,pgresuser,pgrespwd,buildingsnapshotid):
	conn_string = "host='"+str(host)+"' dbname='"+str(pgresdbname)+"' user='"+str(pgresuser)+"' password='"+str(pgrespwd)+"'"
	try:
		conn = psycopg2.connect(conn_string)
	except psycopg2.Error as e:
		print "Unable to connect to database"
		print e.pgerror
		print e.diag.message_detail
		sys.exit(1)
	cur = conn.cursor()

        #retrieve energy details from seed_timeseries for building using buildingsnapshot_id
	cur.execute('SELECT to_char(min(seed_timeseries.begin_time),\'DD Mon YYYY HH12:MI:SS\'),to_char(max(seed_timeseries.end_time),\'DD Mon YYYY HH12:MI:SS\') FROM seed_timeseries,seed_meter,seed_meter_canonical_building,seed_buildingsnapshot WHERE seed_timeseries.meter_id = seed_meter.id AND seed_meter_canonical_building.meter_id = seed_meter.id AND seed_buildingsnapshot.canonical_building_id = seed_meter_canonical_building.canonicalbuilding_id AND seed_buildingsnapshot.id = %(param_bs)s',{'param_bs': buildingsnapshotid})

        rows = cur.fetchall()
	minMaxDateArray = []
        #insert in seed_timeseries
        for row in rows:
                minMaxDateArray.append(row[0])
		minMaxDateArray.append(row[1])
        else:
                return minMaxDateArray

def getMonthlyAggData(host,pgresdbname,pgresuser,pgrespwd,buildingsnapshotid):
        conn_string = "host='"+str(host)+"' dbname='"+str(pgresdbname)+"' user='"+str(pgresuser)+"' password='"+str(pgrespwd)+"'"
        try:
                conn = psycopg2.connect(conn_string)
        except psycopg2.Error as e:
                print "Unable to connect to database"
                print e.pgerror
                print e.diag.message_detail
                sys.exit(1)
        cur = conn.cursor()

        #retrieve energy details from seed_timeseries for building using buildingsnapshot_id
        cur.execute('SELECT to_char(seed_timeseries.begin_time,\'Mon-YYYY\'),to_char(seed_timeseries.begin_time,\'DD Mon YYYY HH12:MI:SS\'),to_char(seed_timeseries.end_time,\'DD Mon YYYY HH12:MI:SS\'),seed_meter.energy_type,SUM(seed_timeseries.reading) FROM seed_timeseries,seed_meter,seed_meter_canonical_building,seed_buildingsnapshot WHERE seed_timeseries.meter_id = seed_meter.id AND seed_meter_canonical_building.meter_id = seed_meter.id AND seed_buildingsnapshot.canonical_building_id = seed_meter_canonical_building.canonicalbuilding_id AND seed_buildingsnapshot.id = %(param_bs)s group by seed_meter.energy_type,seed_timeseries.begin_time,seed_timeseries.end_time order by seed_timeseries.begin_time, seed_meter.energy_type',{'param_bs': buildingsnapshotid})

        rows = cur.fetchall()
	dataDictArr = []
        #insert in seed_timeseries
        for row in rows:
		dataDict = {}
                dataDict['month_year']=row[0]
		dataDict['begin_time']=row[1]
		dataDict['end_time']=row[2]
		if row[3]==0:
			dataDict['energy_type']='electric'
		elif row[3]==1:
			dataDict['energy_type']='gas'
		dataDict['reading']=str(row[4])
		dataDictArr.append(dataDict)
        else:
		#print dataDictArr
                return dataDictArr

def createJSON(monthlyAggregatedData, monthlyCDDHDDArray, bldgDetails):
	jsonElementDict = {}
	if len(bldgDetails) > 2:
		jsonElementDict['city'] = bldgDetails[0]
		jsonElementDict['state'] = bldgDetails[1]
		if bldgDetails[2] is None:
			jsonElementDict['gross_floor_area'] = ''
		else:
			jsonElementDict['gross_floor_area'] = bldgDetails[2]
	energyDataArray = []

	for num in range(0,len(monthlyAggregatedData)):
		if monthlyCDDHDDArray.get(monthlyAggregatedData[num].get('month_year')) is not None:
			monthlyAggregatedData[num]['average_monthly_temperature'] =  monthlyCDDHDDArray.get(monthlyAggregatedData[num].get('month_year')).split(':')[0]
			monthlyAggregatedData[num]['cdd'] =  monthlyCDDHDDArray.get(monthlyAggregatedData[num].get('month_year')).split(':')[1]
			monthlyAggregatedData[num]['hdd'] =  monthlyCDDHDDArray.get(monthlyAggregatedData[num].get('month_year')).split(':')[2]
	jsonElementDict['energy_data'] = monthlyAggregatedData
	return jsonElementDict

#if __name__=='__main__':
def getJSONResponse(buildingid):
	host = '128.2.110.179'
	pgresdbname = 'seed-deploy'
	pgresuser = 'seed-admin'
	pgrespwd = 'SEEDDB@architecture.cmu.edu'
	localtimezone = 'America/New_York'
	buildingsnapshotid = buildingid
	baseTempCdd = '65.0'
	baseTempHdd = '55.0'
	monthlyAggregatedData = getMonthlyAggData(host,pgresdbname,pgresuser,pgrespwd,buildingsnapshotid)
	minMaxDateArray = getMinMaxDates(host,pgresdbname,pgresuser,pgrespwd,buildingsnapshotid)
        bldgDetails = getBuildingDetails(host,pgresdbname,pgresuser,pgrespwd,buildingsnapshotid)
	if bldgDetails[2] is None:
		print len(bldgDetails)
	if(len(minMaxDateArray) == 2 and len(bldgDetails)>=2):
		cddStart = datetime.datetime.now()
                monthlyCDDHDDArray1 = getCDDHDD(minMaxDateArray[0],minMaxDateArray[1],bldgDetails[0],bldgDetails[1],baseTempCdd,baseTempHdd)
                cddEnd = datetime.datetime.now()
		print (cddEnd - cddStart)
		if(monthlyCDDHDDArray1 != 'NA'):
			outputJSON = createJSON(monthlyAggregatedData, monthlyCDDHDDArray1, bldgDetails)
		else:
			outputJSON = '{errorMsg=\'Couldn\'t retrieve station and temperatue info\'}'
	else:
		outputJSON = '{errorMsg=\'Couldn\'t find building info\'}'
	return outputJSON
