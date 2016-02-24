# The process starts from the /PMinput/ folder
# Assuming PMinput contains valid PM table downloaded from EnergyStar Website
import os

import template2json as tm
import postProcess as ps
import PM2template as pm

## ## ## ## ## ## ## ## ## ## ##
## logging and debugging logger.info settings
import logging

logger = logging.Logger('mainroutine')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# read PM excel to template excel
logger.info('\n## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##')
logger.info('converting pm to template')
logger.info('## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##')
pm_in_dir = os.getcwd() + "/PMfile/PMinput/"
pm_out_dir = os.getcwd() + "/PMfile/PMoutput/"
pm.readPM(pm_in_dir, pm_out_dir)
logger.info('\nend converting pm to template')

pm_out_dir = os.getcwd() + "/PMfile/PMoutput/"
logger.info('\n## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##')
logger.info('process template')
logger.info('## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##')
# process template excel
json_in_dir = pm_out_dir
#json_out_dir = os.getcwd() + "/Jsonfile/raw/"
json_out_dir = os.getcwd() + "/Jsonfile/local_raw/"
# query goes here
tm.pm2json(json_in_dir, json_out_dir)

logger.info('\n## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##')
logger.info('post process json')
logger.info('## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##')
# post-process json files
#post_out_dir = os.getcwd() + "/Jsonfile/post/" #original file, there's an offset
post_out_dir = os.getcwd() + "/Jsonfile/local_post/"
ps.postProcess(json_out_dir, post_out_dir)
logger.info('\n## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##')
logger.info('finished')
logger.info('## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##')
