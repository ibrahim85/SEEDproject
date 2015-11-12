import requests
import GreenButtonParser as parser
import xml.etree.ElementTree as ET

# retrieve GreenButton XML data through url
def __request_green_button_by_url(url):
    response = requests.get(url, verify=False)
    if(response.status_code == 200):
        print 'Get GreenButton XML file successfully'
        xml_data = response.text
		
		# xml_data is just text, this step parse the text into XML data structure
        root = ET.fromstring(xml_data)
        return root
    else:
        print 'Request GreenButton Data Error '.response.status_code
        return None

# path: file path to GreenButton XML data file, this function
# may be used later for GreenButton XML data file upload support
def __request_green_button_by_file(path):
    root = ET.parse(path).getroot()
    return root

# this function will be called outside this file
def get_gb_data(url, building_id):
    root = __request_green_button_by_url(url)
    
    if root==None:
        return None
 
	# call data parser function to get parsed timeseries data
    ts_data = parser.gb_xml_parser(root, building_id)
	
    return ts_data
