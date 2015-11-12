import json
import requests
import ConfigParser

# insert more than one records into Karios
def batch_insert_kairosdb(meta_data, ts_data):
    wrap = []
    db_data = ConfigParser.parse_config('./KairosConfig.conf')

	# extract timeseries data from input
    for ts, meta in zip(ts_data, meta_data):
        insert_data = {}
        insert_data['name'] = db_data['measurement']
        insert_data['timestamp'] = ts[0]+'000'
        insert_data['type'] = 'double'
        insert_data['value'] = ts[2]
        insert_data['tags'] = meta      # set meta_data as tag for this record
        
        wrap.append(insert_data)
    
	# convert to JSON format
    json_insert_data = json.dumps(wrap)
    
	# do the insert
    r = requests.post(db_data['insert_url'], data=json_insert_data)
    
	# report error if any
    if r.status_code!=204:
        print 'Error '+str(r.status_code)
        print r.text

# gb_data: parsed GreenButton data, output of parser->analyser->
def insert(gb_data):
    #assemble insert data
    meta_list = []
    ts_list = []
    
    for ts_cell in gb_data:
        meta_data = {}
        meta_data['building_snapshot_id'] = ts_cell['building_snapshot_id']
        meta_data['usage_point_id'] = ts_cell['usage_point_id']
        meta_data['meter_id'] = ts_cell['meter_id']
        meta_data['interval_block_id'] = ts_cell['interval_block_id']
        meta_data['usage_kind'] = ts_cell['usage_kind']
        meta_data['reading_kind'] = ts_cell['reading_kind']
        meta_data['tens'] = ts_cell['tens']
        meta_data['uom'] = ts_cell['uom']
		meta_data['interval'] = ts_cell['interval']
        meta_list.append(meta_data)
        
        ts_data = [ts_cell['start'], ts_cell['interval'], ts_cell['value']]
        ts_list.append(ts_data)
    
    # do insert
    batch_insert_kairosdb(meta_list, ts_list)
