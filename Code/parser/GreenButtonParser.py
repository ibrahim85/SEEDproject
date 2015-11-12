global root_ns
global gb_ns

root_ns = '{http://www.w3.org/2005/Atom}'
gb_ns = '{http://naesb.org/espi}'

# get the text from beginning to the first / sign, if link starts with /, then second / counts
# e.g. for '/www.cmu.edu/soarch/iw', 'www.cmu.edu' will be returned
def __get_link_href_head(link):
    href = link.get('href')
    if href[0:1]=='/':
        href = href[1:]
    first_slash = href.find('/')
    return href[0:first_slash]

# get the text from the last / sign to the end
# e.g. for '/www.cmu.edu/soarch/iw', 'iw' will be returned
def __get_link_href_tail(link):
    href = link.get('href')
    last_slash = href.rfind('/')
    return href[(last_slash+1):]

# return {usage kind id}
# XML structure:
# <entry>
# 	<content type="text/XML">
# 		<UsagePoint xmlns="http://naesb.org/espi">
# 			<ServiceCategory>
# 				<kind>0</kind>
# 			</ServiceCategory>
# 		</UsagePoint>
# 	</content>
# </entry>
def __parse_usage(up_type):
    sc = up_type.find(root_ns+'content').find(gb_ns+'UsagePoint').find(gb_ns+'ServiceCategory')
    
    sc_info = {}
    sc_info['kind'] = sc.find(gb_ns+'kind').text
	
    return sc_info

# Return {reading kind, tens, uom}
# XML structure:
# <entry>
# 	<content type="text/XML">
# 		<ReadingType xmlns="http://naesb.org/espi">
# 			<accumulationBehaviour>4</accumulationBehaviour>
# 			<commodity>1</commodity>
# 			<dataQualifier>12</dataQualifier>
# 			<defaultQuality>0</defaultQuality>
# 			<flowDirection>1</flowDirection>
# 			<intervalLength>900</intervalLength>
# 			<kind>12</kind>
# 			<phase>769</phase>
# 			<powerOfTenMultiplier>0</powerOfTenMultiplier>
# 			<timeAttribute>0</timeAttribute>
# 			<uom>72</uom>
# 		</ReadingType>
# </entry>
def __parse_reading_type(rt_entry):
    rt = rt_entry.find(root_ns+'content').find(gb_ns+'ReadingType')
    
    rt_info = {}
    rt_info['kind'] = rt.find(gb_ns+'kind').text
    rt_info['tens'] = rt.find(gb_ns+'powerOfTenMultiplier').text
    rt_info['uom'] = rt.find(gb_ns+'uom').text
    return rt_info

# mr_entry: meter reading entry
# Return corresponding ReadingType element id or NULL if failed
def __get_reading_type_id_of_meter(mr_entry):
    links = mr_entry.findall(root_ns+'link[@rel="related"]')
    for link in links:
        head = __get_link_href_head(link)
        if head == 'ReadingType':
            return link.get('href')
    return null

# Return {ts_start, value}
# XML structure
# <IntervalReading>
# 	<timePeriod>
# 		<duration>900</duration>
# 		<start>1441080000</start>
# 	</timePeriod>
# 	<value>0</value>
# </IntervalReading>
def __parse_interval_reading(reading):
    time_period = reading.find(gb_ns+'timePeriod')
    
    res = {}
    res['ts_start'] = time_period.find(gb_ns+'start').text
    res['interval'] = time_period.find(gb_ns+'duration').text
    res['value'] = reading.find(gb_ns+'value').text
    
    return res;

# Replace / with _, since key doesn't allow / character
def __clean_up_for_key(key):
    return string.replace(key, '/', '_')

# function called outside
def gb_xml_parser(root, building_snapshot_id):
    entries = root.findall(root_ns+'entry')
    
	# relationship maps
    usage_info_map = {}
    reading_type_map = {}
    interval_block_array = []
    
    # Sequence requirements: UsagePoint->MeterReading->IntervalBlock
    for entry in entries:
        up_link = entry.find(root_ns+'link[@rel="up"]')
        self_link = entry.find(root_ns+'link[@rel="self"]')
        
        type = __get_link_href_tail(up_link)
        if type == 'UsagePoint':
            usage_point_id = __get_link_href_tail(self_link)
            
            # Parse UsagePoint, get service category kind
            usage_info = __parse_usage(entry)            
            usage_info_map[usage_point_id] = usage_info
        elif type == 'MeterReading':
            meter_id = __get_link_href_tail(self_link)
            
            # Get corresponding ReadingType
            ref_reading_type_id = __get_reading_type_id_of_meter(entry)
        elif type == 'IntervalBlock':
            interval_block_id = __get_link_href_tail(self_link)
            
			# link interval_block with usage_point, meter, and reading type
            interval_block_cell = {}
            interval_block_cell['usage_point_id'] = usage_point_id
            interval_block_cell['meter_id'] = meter_id
            interval_block_cell['ref_reading_type_id'] = ref_reading_type_id
            interval_block_cell['interval_block_id'] = interval_block_id
            interval_block_cell['element'] = entry
            
            interval_block_array.append(interval_block_cell)
        elif type == 'ReadingType':
            reading_type_id = self_link.get('href')
            
            # Parse ReadingType
            reading_type_info = __parse_reading_type(entry)
            reading_type_map[reading_type_id] = reading_type_info
            
    # Extract data from interval blocks
    ts_data = []
    for interval_block_cell in interval_block_array:
        usage_point_id = interval_block_cell['usage_point_id']
        meter_id = interval_block_cell['meter_id']
        interval_block_id = interval_block_cell['interval_block_id']
        ref_reading_type_id = interval_block_cell['ref_reading_type_id']
		
        entry = interval_block_cell['element']
        
		# retrieve usage point and reading type information
        usage_info = usage_info_map[usage_point_id]
        reading_info = reading_type_map[ref_reading_type_id]
        
		# get all <IntervalBlock> elements
        blocks = entry.find(root_ns+'content').findall(gb_ns+'IntervalBlock')
        for block in blocks:
			# get all <IntervalReading> elements
            interval_readings = block.findall(gb_ns+'IntervalReading')
			
            for interval_reading in interval_readings:
                interval_reading_info = __parse_interval_reading(interval_reading)
                
                ts_cell = {}
                ts_cell['start'] = interval_reading_info['ts_start']
                ts_cell['value'] = interval_reading_info['value']
                ts_cell['interval'] = interval_reading_info['interval']
                ts_cell['building_snapshot_id'] = building_snapshot_id
                ts_cell['usage_point_id'] = usage_point_id
                ts_cell['meter_id'] = meter_id
                ts_cell['interval_block_id'] = interval_block_id
                # TODO add new data here
                ts_cell['usage_kind'] = usage_info['kind']
                ts_cell['reading_kind'] = reading_info['kind']
                ts_cell['tens'] = reading_info['tens']
                ts_cell['uom'] = reading_info['uom']
                
				# put timeseries data into to be returned collection
                ts_data.append(ts_cell)
    
    return ts_data
