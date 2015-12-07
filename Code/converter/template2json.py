# convert excel template to json file
# change from converter.py:
# reading input template is template10field.xlsx
import os
import pandas as pd
import json
import glob
import numpy as np

# replace the following directory name and filename
def pm2json(in_dir, out_dir):
    filelist = glob.glob(in_dir + "*.xlsx")
    for excel in filelist:
        pm2jsonSingle(excel, in_dir, out_dir)

def pm2jsonSingle(excel, in_dir, out_dir):
    meter_con_df = pd.read_excel(excel, sheetname=0)
    #print meter_con_df.ix[:10]

    ## ## ## ## ## ## ## ## ## ## ##
    ## query code goes here
    ## ## ## ## ## ## ## ## ## ## ##
    # address_dict: key:address, null as its initial value
    ar = meter_con_df['Street Address'].values
    address_dict = {}
    for a in np.nditer(ar, flags=['refs_ok']):
        address_dict['{0}'.format(a)] = None
    # uncomment this when the query code is working
    # meter_con_df['buildingsnapshot_id'] = address_dict['Street Address'].map(lambda x:address_dict[x['buildingsnapshot_id'])
    # meter_con_df['canonical_id'] = address_dict['Street Address'].map(lambda x:address_dict[x['canonical_building'])
    ## ## ## ## ## ## ## ## ## ## ##
    ## query code ends
    ## ## ## ## ## ## ## ## ## ## ##

    # Calculate time interval of days
    meter_con_df['interval'] = meter_con_df['End Date'] - meter_con_df['Start Date']
    meter_con_df['reading_kind'] = 'energy'

    # renaming columns of df
    name_lookup = {u'Start Date':u'start',
                   u'End Date':u'end',
                   u'Portfolio Manager Meter ID':u'meter_id',
                   u'Usage/Quantity':u'value',
                   u'Usage Units':'uom'}

    meter_con_df = meter_con_df.rename(columns=name_lookup)

    file_out = excel[len(in_dir):-5] + '_json.txt'
    # the 'interval' output is in [ns], so use ns for all time object and
    # post process with postProcess.py
    meter_con_df.to_json(out_dir + file_out, 'records',
                         date_unit = 'ns')
