# read a folder containing excel files downloaded from EnergyStar PM
import os
import pandas as pd
import glob

## ## ## ## ## ## ## ## ## ## ##
## create logger for print ##
## ## ## ## ## ## ## ## ## ## ##
import logging
logger = logging.Logger('mainroutine')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def readPM(in_dir, out_dir):
    in_dir = os.getcwd() + "/PMfile/PMinput/"
    out_dir = os.getcwd() + "/PMfile/PMoutput/"
    filelist = glob.glob(in_dir + "*.xlsx")
    for excel in filelist:
        processOneFile(excel, in_dir, out_dir)

def processOneFile(filename, in_dir, out_dir):
    logger.info('convert file{0} to template'.format(filename))
    df_address = pd.read_excel(filename, sheetname=0, skiprows=4, header=5,
                               parse_cols = [1, 2])
    address_dict = dict(zip(df_address['Portfolio Manager ID'],
                           df_address['Street Address']))
    df_energy = pd.read_excel(filename, sheetname=5, skiprows=4, header=5,
                              parse_cols = [0, 1, 2, 4, 6, 7, 9, 10, 11])
    df_energy.insert(0, 'Street Address',
                     df_energy['Portfolio Manager ID'].map(lambda x:
                                                           address_dict[x]))
    df_energy.info()
    df_energy.rename(columns={'Portfolio Manager ID': 'Custom ID',
                              'Portfolio Manager Meter ID':
                              'Custom Meter ID'},
                     inplace=True)
    file_out = filename[len(in_dir):-5] + '_temp.xlsx'
    df_energy.to_excel(out_dir + file_out, index=False)

#readPM()
