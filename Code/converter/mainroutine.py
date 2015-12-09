import os
import template2json as tm
import postProcess as ps
import PM2template as pm

## ## ## ## ## ## ## ## ## ## ##
## logging and debugging logger.info settings
import getopt
import logging
import sys

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

logger.info('\n## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##')
logger.info('process template')
logger.info('## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##')
# process template excel
json_in_dir = pm_out_dir
json_out_dir = os.getcwd() + "/Jsonfile/raw/"
# query goes here
tm.pm2json(json_in_dir, json_out_dir)

logger.info('\n## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##')
logger.info('post process json')
logger.info('## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##')
# post-process json files
post_out_dir = os.getcwd() + "/Jsonfile/post/"
ps.postProcess(json_out_dir, post_out_dir)
logger.info('\n## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##')
logger.info('finished')
logger.info('## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##')
