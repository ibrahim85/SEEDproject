# separate excel sheets to csv files
import pandas as pd
import os
import glob
import datetime
import numpy as np

## ## ## ## ## ## ## ## ## ## ##
## logging and debugging logger.info settings
import logging
import sys

logger = logging.Logger('reading')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

# read the ith sheet of excel and write a csv to outdir
def excel2csv_single(excel, i, outdir):
    df = pd.read_excel(excel, sheetname=int(i), skiprows=4, header=5)
    file_out = outdir + 'sheet-{0}-all_col'.format(i) + '.csv'
    df.to_csv(file_out, index=False)

# read static info to a data frame
# sheet-0: static info
#          1. postal code : take first 5 digit
#          2. property name : take the substring before '-'
def read_static():
    indir = os.getcwd() + '/csv/select_column/'
    csv = indir + 'sheet-0.csv'
    logger.debug('read static info')
    df = pd.read_csv(csv)
    # take the five digits of zip code
    df['Property Name'] = df['Property Name'].map(lambda x: x[:x.find('-')])
    df['Postal Code'] = df['Postal Code'].map(lambda x: x[:5])
    #logger.debug(df[:20])
    return df

def split_energy_building():
    indir = os.getcwd() + '/csv/select_column/'
    csv = indir + 'sheet-5.csv'
    logger.debug('split energy to building')
    outdir = os.getcwd() + '/csv/single_building/'
    df = pd.read_csv(csv)
    # auto-fill missing data
    df = df.fillna(0)
    group_building = df.groupby('Portfolio Manager ID')
    for name, group in group_building:
        group.to_csv(outdir + 'pm-' + str(name) + '.csv', index=False)

def check_null(csv):
    print 'checking number of missing values for columns'
    df = pd.read_csv(csv)
    for col in df:
        print '## ------------------------------------------##'
        print col
        df_check = df[col].isnull()
        df_check = df_check.map(lambda x: 'Null' if x else 'non_Null')
        print df_check.value_counts()

def check_null_df(df):
    print 'checking number of missing values for columns'
    for col in df:
        print '## ------------------------------------------##'
        print col
        df_check = df[col].isnull()
        df_check = df_check.map(lambda x: 'Null' if x else 'non_Null')
        print df_check.value_counts()

def get_range(df):
    print 'range for columns'
    for col in df:
        if not (col == 'Meter Type' or col == 'Usage Units'):
            print '{0:>28} {1:>25} {2:>25}'.format(col, df[col].min(),
                                               df[col].max())

def count_nn(df, col):
    df['is_nn'] = df[col].map(lambda x: '>=0' if x >= 0 else '<0')
    series = df['is_nn'].value_counts()
    print series
    grouped = df.groupby(['is_nn', 'Meter Type'])
    print grouped.size()
    '''
    for name, group in grouped:
        print name
        print group
    '''
    df.drop('is_nn', axis = 1, inplace=True)

def format_building():
    indir = os.getcwd() + '/csv/single_building/'
    filelist = glob.glob(indir + '*.csv')
    outdir = os.getcwd() + '/csv/single_building_allinfo/'
    for csv in filelist:
        filename = csv[csv.find('pm'):]
        logger.info('format file: {0}'.format(filename))
        df = pd.read_csv(csv)
        df.set_index
        # create year and month column
        group_type = df.groupby('Meter Type')

        for name, group in group_type:
            outfilename = filename[:-4] + '-' + str(name) + '.csv'
            outfilename = outfilename.replace('/', 'or')
            outfilename = outfilename.replace(':', '-')
            print outfilename
            group.to_csv(outdir + outfilename, index=False)
format_building()
