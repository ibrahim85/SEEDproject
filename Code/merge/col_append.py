import os
import glob
import pandas as pd
import reading as rd

## ## ## ## ## ## ## ## ## ## ##
## logging and debugging logger.info settings
import logging
import sys

logger = logging.Logger('reading')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def main():
    df = pd.read_csv(os.getcwd() + '/csv/cleaned/all_info.csv')
    df.info()
    rd.check_null_df(df)
    grouped = df.groupby('Meter Type')
    #print grouped.groups.keys()

    '''
    df_01 = grouped.get_group('Electric - Grid')
    #df_01.info()
    #df_01.set_index(['End Date', 'Portfolio Manager ID'], inplace=True)
    df_01.info()
    df_merge = pd.merge(df, df_01, how='left', on=['End Date', 'Portfolio Manager ID'])
    df_merge.info()


    df_02 = grouped.get_group('Natural Gas')
    df_02.info()

    df_merge = pd.merge(df_01, df_02, how='outer', on=['End Date', 'Portfolio Manager ID'])
    df_el_gs = df_merge
    df_el_gs.info()
    df_03 = grouped.get_group('Fuel Oil (No. 2)')
    df_04 = grouped.get_group('Potable: Mixed Indoor/Outdoor')
    '''


main()

