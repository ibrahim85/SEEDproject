# convert excel template to json file
# change from converter.py:
# reading input template is template10field.xlsx
import os
import pandas as pd
import json
import glob
import numpy as np
import datetime as dt
import random as rd

import query as qr

## ## ## ## ## ## ## ## ## ## ##
## create logger for print ##
## ## ## ## ## ## ## ## ## ## ##
import logging
logger = logging.Logger('mainroutine')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

## ## ## ## ## ## ## ## ## ## ##
## main function ##
## ## ## ## ## ## ## ## ## ## ##
# replace the following directory name and filename
def pm2json(in_dir, out_dir):
    filelist = glob.glob(in_dir + "*.xlsx")
    for excel in filelist:
        pm2jsonSingle(excel, in_dir, out_dir)

# There are no documentation for whether PM is in local or UTC time
# Assume it is in local time
# Solution: offset time to 12:00 noon
# Why: the date do not change for converting to UTC
# Reasoning: according to: http://www.timetemperature.com/abbreviations/united_states_time_zone_abbreviations.shtml, there are no UTC-local difference of over 11 hours, so after the conversion, the time shown in SEED do not change its date.

def pm2jsonSingle(excel, in_dir, out_dir):
    logger.info('\ntemplate to json:{0}'.format(excel))
    meter_con_df = pd.read_excel(excel, sheetname=0)
    logger.debug(meter_con_df.ix[:10])

    logger.info('start query')
    ar = meter_con_df['Street Address'].values
    address_dict = {}
    for a in np.nditer(ar, flags=['refs_ok']):
        address_dict['{0}'.format(a)] = None
    qr.retrieveID(address_dict)
    logger.debug('address_dict: {0}'.format(address_dict))

    meter_con_df['buildingsnapshot_id'] = meter_con_df['Street Address'].map(lambda x:address_dict[x]['buildingsnapshot_id'])
    meter_con_df['canonical_id'] = meter_con_df['Street Address'].map(lambda x:address_dict[x]['canonical_building'])
    #logger.debug(meter_con_df.info())
    logger.info('end query')

    # Calculate time interval of days
    meter_con_df['interval'] = meter_con_df['End Date'] - meter_con_df['Start Date']

    meter_con_df['End Date'] = meter_con_df['End Date'].map(lambda x: x+dt.timedelta(hours=12))
    meter_con_df['Start Date'] = meter_con_df['Start Date'].map(lambda x: x+dt.timedelta(hours=12))

    meter_con_df['reading_kind'] = 'energy'

    # fill Custom ID and Meter ID with random number !!update Dec 18, 2015
    meter_con_df['Custom ID'].fillna(str(rd.randint(1, 10)), inplace=True)
    meter_con_df['Custom Meter ID'] = meter_con_df.apply(lambda row:row['Meter Type'] if np.isnan(row['Custom Meter ID']) else row['Custom Meter ID'], axis=1)

    # renaming columns of df
    name_lookup = {u'Start Date':u'start',
                   u'End Date':u'end',
                   u'Custom Meter ID':u'meter_id',
                   u'Usage/Quantity':u'value',
                   u'Usage Units':'uom'}

    meter_con_df = meter_con_df.rename(columns=name_lookup)
    meter_con_df.info()

    file_out = excel[len(in_dir):-5] + '_json.txt'
    # the 'interval' output is in [ns], so use ns for all time object and
    # post process with postProcess.py
    meter_con_df.to_json(out_dir + file_out, 'records',
                         date_unit = 'ns')
