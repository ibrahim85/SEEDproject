import os
import pandas as pd
import json

# replace the following directory name and filename
read_dir_name = os.getcwd() + "/PMfile/"
read_file_name = "OSHERextension.xlsx"
write_dir_name = os.getcwd() + "/Jsonfile/"
write_file_name = "test.txt"

meter_con_df = pd.read_excel(read_dir_name + read_file_name, sheetname='Meter Consumption Data', skiprows=4, header = 5)

d = meter_con_df.to_json(write_dir_name + write_file_name, 'records', date_unit = 's')
#json.dump(d, open(write_dir_name + write_file_name, 'w'))

# TO DOs
# df header name conversion from PM to postgreSQL
