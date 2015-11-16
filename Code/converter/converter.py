import os
import pandas as pd
import json

# replace the following directory name and filename
read_dir_name = os.getcwd() + "/PMfile/"
read_file_name = "OSHERextension.xlsx"
write_dir_name = os.getcwd() + "/Jsonfile/"
write_file_name = "test.txt"

meter_con_df = pd.read_excel(read_dir_name + read_file_name, sheetname='Meter Consumption Data', skiprows=4, header = 5)

# Calculate time interval of days
meter_con_df['interval'] = meter_con_df['End Date'] - meter_con_df['Start Date']
meter_con_df['reading_kind'] = 'energy'

# renaming columns of df
name_lookup = {u'Start Date':u'start',
               u'End Date':u'End Date',
               u'Usage/Quantity':u'value',
               u'Usage Units':'uom'}

meter_con_df = meter_con_df.rename(columns=name_lookup)

# the 'interval' output is in [ns]
d = meter_con_df.to_json(write_dir_name + write_file_name, 'records', date_unit = 's')

# TO DOs
