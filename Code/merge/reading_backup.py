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

def clear_data():
    indir = os.getcwd() + '/csv/select_column/'
    # check null value
    '''
    filelist = glob.glob(indir + '*.csv')
    for csv in filelist:
        check_null(csv)
    '''

    # return range of values
    df = pd.read_csv(indir + 'sheet-5.csv')
    get_range(df)

    # count non-neg value for column
    count_nn(df, 'Usage/Quantity')

    # discard null 'End Date' value and negative 'Usage/Quantity'
    # fill empty cost with -1 for current use
    logger.debug('Null value count of \'Cost ($)\' before fillna')
    print df['Cost ($)'].isnull().value_counts()
    logger.debug('Fill \'Cost ($)\' with -1')
    df['Cost ($)'].fillna(-1, inplace=True)
    logger.debug('Null value count of \'Cost ($)\' after fillna')
    print df['Cost ($)'].isnull().value_counts()

    logger.debug('Null value count of \'End Date\' before drop null')
    print df['End Date'].isnull().value_counts()
    logger.debug('Drop null value of \'End Date\'')
    df.dropna(inplace=True)
    logger.debug('Null value count of \'End Date\'after drop null')
    logger.debug(df['End Date'].isnull().value_counts())

    logger.debug('negative value count of \'Usage\'')
    df['sign_nn'] = df['Usage/Quantity'].map(lambda x: '>=0' if x >= 0 else '<0')
    logger.debug(df['sign_nn'].value_counts())
    logger.debug('Mark negative value as nan')
    df['mark_nn'] = df['Usage/Quantity'].map(lambda x: x if x >= 0 else np.nan)
    logger.debug(df['mark_nn'].isnull().value_counts())
    df.dropna(inplace=True)
    logger.debug('Null value count of removing negative usage')
    logger.debug(df['Usage/Quantity'].isnull().value_counts())

    # drop temporary columns
    df.drop(['sign_nn', 'mark_nn'], axis = 1, inplace=True)
    # return range of column after removing illegal values
    get_range(df)

    # count non-neg value for column
    logger.debug('Checking non-negativity after initial clean')
    count_nn(df, 'Usage/Quantity')

    # create 'Year' and 'Month' column
    df['Year'] = df['End Date'].map(lambda x : x[:4])
    df['Month'] = df['End Date'].map(lambda x: x[5:7])

    logger.debug('Final range of the data')
    get_range(df)

    return df



    '''
    print 'group by building and meter type:'
    group_building = df.groupby(['Portfolio Manager ID', 'Meter Type'])
    print group_building.size()
    group_type = df.groupby('Meter Type')
    print 'number of meter type: {0}'.format(len(group_type.groups))
    print group_type.size()
    group_type = df.groupby('Year')
    print 'number of year: {0}'.format(len(group_type.groups))
    print group_type.size()
    group_type = df.groupby('Property Manager ID')
    print 'number of building: {0}'.format(len(group_type.groups))
    print group_type.size()
    '''

def format_building(df_static):
    excel2csv_single(excel, i, outdir)

    indir = os.getcwd() + '/csv/single_building/'
    filelist = glob.glob(indir + '*.csv')
    outdir = os.getcwd() + '/csv/single_building_allinfo/'
    for csv in filelist:
        filename = csv[csv.find('pm'):]
        logger.info('format file: {0}'.format(filename))
        df = pd.read_csv(csv)
        # create year and month column
        df['Year'] = df['End Date'].map(lambda x: x[:4])
        df['Month'] = df['End Date'].map(lambda x: x[5:7])
        df.drop('End Date', 1, inplace=True)
        group_type = df.groupby('Meter Type')

        for name, group in group_type:
            outfilename = filename[:-4] + '-' + str(name) + '.csv'
            outfilename = outfilename.replace('/', 'or')
            outfilename = outfilename.replace(':', '-')
            print outfilename
            group.to_csv(outdir + outfilename, index=False)
        '''
        df_base = group_type.first()
        acclist = [df_base]
        for name, group in group_type:
            acc = acclist.pop()
            acclist.append(pd.merge(acc, group, how='inner', on=['Year', 'Month']))
        print acclist.pop()

        # default to water
        df_base = group_type.get_group('Potable: Mixed Indoor/Outdoor')
        if 'Electric - Grid' in group_type.groups:
            merge_1 = pd.merge(df_base, group_type.get_group('Electric - Grid'), how = 'right', on = ['Year', 'Month'], suffixes = ['_water', '_elec'])
        else:
            merge_1 = df_base
        #logger.debug(merge_1[:10])

        #if 'Natural Gas' in group_type.groups:
        gas = group_type.get_group('Natural Gas')
        merge_2 = pd.merge(gas, group_type.get_group('Fuel Oil (No. 2)'), how = 'right', on = ['Year', 'Month'], suffixes = ['_gas', '_oil'])
        logger.debug(merge_2[:10])
        merge_all = pd.merge(merge_1, merge_2, left_on = ['Year_elec', 'Month_elec'], right_on = ['Year_gas', 'Month_gas'])
        logger.debug('merged ###########################')
        #logger.debug(merge_3[:10])
        merge_all.info()
        '''


        '''
        df_all.drop('Usage/Quantity')
        logger.debug(df_all[:10])

        df_join = df_all.join(df_static, on = 'Portfolio Manager ID',
                              lsuffix = 'l', rsuffix = '_r')
        df_join.drop('Portfolio Manager ID_l', 1)
        df_join.drop('Portfolio Manager ID_r', 1)
        logger.debug(df[:10])
        outfilename = outdir + 'post-' + filename
        df_join.to_csv(outfilename, index=False)
        '''

# process all excels
def main():
    '''
    indir = os.getcwd() + '/input/'
    filelist = glob.glob(indir + '*.xlsx')
    logger.info('separate excel file to csv: {0}'.format(filelist))
    outdir = os.getcwd() + '/csv/all_column/'
    for excel in filelist:
        for i in ['0', '5']:
            logger.info('reading sheet {0}'.format(i))
            excel2csv_single(excel, i, outdir)

    logger.info('read csv in {0} with selected column: {1}'.format(outdir,
                                                                   filelist))
    filelist = glob.glob(os.getcwd() + '/csv/all_column/' + '*.csv')
    col_dict = {'0':[0, 1, 5, 7, 9, 12], '5':[1, 2, 4, 6, 8, 9, 10]}
    for csv in filelist:
        filename = csv[csv.find('sheet'):]
        logger.info('reading csv file: {0}'.format(filename))
        idx = filename[filename.find('-') + 1:filename.find('-') + 2]
        logger.info('file index: {0}'.format(idx))
        df = pd.read_csv(csv, usecols = col_dict[idx])
        outdir = os.getcwd() + '/csv/select_column/'
        outfilename = filename[:filename.find('all') - 1] + '.csv'
        logger.info('outdir = {0}, outfilename = {1}'.format(outdir,
            outfilename))
        df.to_csv(outdir + outfilename, index=False)

    # split energy data to single building
    split_energy_building()
    # data frame containing static information
    df_static = read_static()

    format_building(df_static)
    '''


    # process:
    # sheet-1: energy info, read in df,
    #          group by Portfolio Manager ID
    #          write to a folder
    #          for each file in the folder
    #              read to df
    #              group by Meter Type
    #              aggregate all meters in one df

    #              convert date time to year and month
    #          write to a different folder

main()
