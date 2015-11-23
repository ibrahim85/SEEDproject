import os
import pandas as pd
import json

# replace the following directory name and filename
read_dir_name = os.getcwd() + "/PMfile/"
read_file_name = "template.xlsx"
read_sheet_name = 0
write_dir_name = os.getcwd() + "/Jsonfile/"
write_file_name = "test.txt"

def pm2json():
    meter_con_df = pd.read_excel(read_dir_name + read_file_name,
                                 sheetname=read_sheet_name, skiprows=0,
                                 header = 0)

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

    # the 'interval' output is in [ns], so use ns for all time object and
    # post process with postProcess.py
    meter_con_df.to_json(write_dir_name + write_file_name, 'records',
                         date_unit = 'ns')

# TO DOs
