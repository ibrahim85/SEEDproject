# path_to_config: file path of parser configuration file
# the configuration file has multiple lines, the format is key=value
def parse_config(path_to_config):
    res = {}
    f = open(path_to_config, 'r')
    for line in f:
        parsed = line.split('=')
		
		# retrieve key and value
        res[parsed[0].strip()] = parsed[1].strip()
    f.close() 
    return res
