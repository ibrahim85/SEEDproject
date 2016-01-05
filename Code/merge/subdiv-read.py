# separate excel sheets to csv files
import pandas as pd
import os
import glob
import datetime
import numpy as np

import util_io as io

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

def unique_value(df):
    df.info()
    logger.debug('number of unique values')
    for col in df:
        logger.debug('{0}: {1}'.format((col), len(df[col].unique())))

# not used, because EUAS has insufficient number of buildings that match
# PM records
def attemptEUASlookup(df):
    # read and clean up EUAS template
    euas_temp = os.getcwd() + '/input/EUAS.csv'
    logger.debug('Read in EUAS region')
    df_t = pd.read_csv(euas_temp, usecols=['Building ID', 'Region'])
    logger.debug('general info of EUAS data frame')
    unique_value(df_t)
    df_t.drop_duplicates(inplace=True)
    logger.debug('general info of EUAS data frame after remove dup')
    unique_value(df_t)

    # checking common records between two tables
    pm_set = set(df['Building ID'].tolist())
    euas_set = set(df_t['Building ID'].tolist())
    common_id_set = pm_set.intersection(euas_set)
    logger.debug('{0} buildings in PM'.format(len(df['Building ID'].unique())))
    logger.debug('{0} buildings in EUAS'.format(len(df_t['Building ID'].unique())))
    logger.debug('{0} common building records between PM and EUAS'.format(len(common_id_set)))

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
    df['Property Name'] = df['Property Name'].map(lambda x: x.partition(' ')[0][:8])
    df['Postal Code'] = df['Postal Code'].map(lambda x: x[:5])
    logger.debug(df['Property Name'].tolist()[:10])

    df.info()
    df.rename(columns={'Property Name' : 'Building ID',
                       'State/Province' : 'State',
                       'Gross Floor Area' : 'GSF'}, inplace=True)
    df.info()
    print df.groupby(['Country', 'State']).size()

    regionfile = os.getcwd() + '/input/stateRegion.csv'
    logger.debug('reading region look up table')
    df_region = pd.read_csv(regionfile, usecols=['State', 'Region'])

    df_set = set(df['State'].tolist())
    region_set = set(df_region['State'].tolist())
    common_state_set = df_set.intersection(region_set)
    logger.debug('{0} states in PM'.format(len(df['State'].unique())))
    logger.debug('{0} states in Region'.format(len(df_region['State'].unique())))
    logger.debug('{0} common state records between PM and Region'.format(len(common_state_set)))

    logger.debug('Mark non-U.S. records as nan')
    df['mark_nu'] = df['Country'].map(lambda x: True if x == 'United States' else np.nan)
    logger.debug(df['mark_nu'].isnull().value_counts())
    df.dropna(inplace=True)
    logger.debug('Null value count of removing non-U.S. countries')
    logger.debug(df['mark_nu'].isnull().value_counts())
    df.drop('mark_nu', axis = 1, inplace=True)

    df = pd.merge(df, df_region, on='State')
    logger.debug('number of un-mapped Region record')
    logger.debug(df['Region'].isnull().value_counts())

    static_info_file = os.getcwd() + '/csv/cleaned/static_info.csv'
    print static_info_file
    df.to_csv(static_info_file, index=False)

# split energy consumption to different files
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

def get_range(df, col):
    print 'range for columns'
    for col in df:
        if not (col == 'Meter Type' or col == 'Usage Units'):
            print '{0:>28} {1:>25} {2:>25}'.format(col, df[col].min(),
                                               df[col].max())
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
    df.drop('is_nn', axis = 1, inplace=True)

def clean_data():
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

    logger.debug('Null value count of \'Start Date\' before drop null')
    print df['End Date'].isnull().value_counts()

    # return range of column after removing illegal values
    get_range(df)

    # count non-neg value for column
    logger.debug('Checking non-negativity after initial clean')
    count_nn(df, 'Usage/Quantity')

    # create 'Year' and 'Month' column
    #df['Year'] = df['End Date'].map(lambda x : x[:4])

    #logger.debug('Final range of the data')
    #get_range(df)
    df['Start Date'] = df['Start Date'].map(lambda x: np.datetime64(x[:10],'D'))
    df['End Date'] = df['End Date'].map(lambda x: np.datetime64(x[:10],'D'))

    energy_info_file = os.getcwd() + '/csv/cleaned/energy_info.csv'
    df.to_csv(energy_info_file, index=False)

    return df

# input PM excel file to single sheets with all columns in original sheet
def input2csv():
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

# retain only columns used in calculation
def select_col():
    filelist = glob.glob(os.getcwd() + '/csv/all_column/' + '*.csv')
    col_dict = {'0':[0, 1, 5, 7, 8, 9, 12], '5':[1, 2, 4, 5, 6, 8, 9, 10]}
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

# process all excels
def main():
    '''
    input2csv()
    select_col()
    '''
    # data frame containing static information
    df_static = read_static()
    df_energy = clean_data()

    '''
    # read from cleaned data
    df_static = pd.read_csv(os.getcwd() + '/csv/cleaned/static_info.csv')
    df_energy = pd.read_csv(os.getcwd() + '/csv/cleaned/energy_info.csv')
    df_merge = pd.merge(df_energy, df_static, on='Portfolio Manager ID')

    logger.debug('Rearrange columns: ')
    cols = df_merge.columns.tolist()
    #logger.debug('original columns: \n{0}'.format(cols))
    newcols = cols[9:] + cols[:9]
    #logger.debug('new columns: \n{0}'.format(newcols))

    df_merge = df_merge[newcols]
    df_merge.drop('End Date', axis=1, inplace=True)

    # checks
    logger.debug('Number of buildings in PM before merging')
    logger.debug(len(df_merge['Building ID'].unique()))
    df_merge.info()
    grouped = df_merge.groupby(['Building ID', 'Year', 'Month', 'Meter Type'])
    for name, group in grouped:
        if len(group) > 1:
            print name

    #check_null_df(df_merge)
    #count_nn(df_merge, 'Usage/Quantity')
    #merged_file = (os.getcwd() + '/csv/cleaned/all_info_c.csv')
    #df_merge.to_csv(merged_file, index=False)

    # -- merging building meter
    # read euas buildings list
    euas = os.getcwd() + '/input/EUAS.csv'
    logger.debug('Read in EUAS region')
    df_t = pd.read_csv(euas, usecols=['Building ID'])
    df_t['Building ID'] = df_t['Building ID'].map(lambda x: x.partition(' ')[0][:8])
    logger.debug('Number of buildings in EUAS')
    logger.debug(len(df_t['Building ID'].unique()))
    euas_set = set(df_t['Building ID'].tolist())
    assert((len(df_t['Building ID'].unique())) == len(euas_set))

    df_base = df_merge.drop(['Usage/Quantity', 'Usage Units', 'Cost ($)', 'Portfolio Manager Meter ID', 'Meter Type'], axis=1, inplace=False)
    df_base.info()
    df_base = df_base.drop_duplicates()
    logger.debug('Number of buildings in PM')
    logger.debug(len(df_base['Building ID'].unique()))
    logger.debug('info of base table to be joined')
    df_base.info()
    logger.debug('number of common buildings before merging')
    logger.debug(len(euas_set.intersection(set(df_base['Building ID'].tolist()))))

    grouped = df_merge.groupby('Meter Type')
    b1 = set(grouped.get_group('Electric - Grid')['Building ID'].tolist())
    b2 = set(grouped.get_group('Natural Gas')['Building ID'].tolist())
    b3 = set(grouped.get_group('Fuel Oil (No. 2)')['Building ID'].tolist())
    b4 = set(grouped.get_group('Potable: Mixed Indoor/Outdoor')['Building ID'].tolist())
    logger.debug('number of buildings with the four major energy source')
    logger.debug(len(b1.union(b2.union(b3.union(b4)))))
    #print grouped.groups.keys()
    df_01 = grouped.get_group('Electric - Grid')
    df_01.rename(columns={'Usage/Quantity':'elec_amt',
                          'Usage Units':'elec_unit',
                          'Cost ($)':'elec_cost',
                          'Portfolio Manager Meter ID':'elec_meter_id'},
                 inplace=True)
    df_01.drop(['Building ID', 'State', 'Country', 'Postal Code',
                'Year Built', 'GSF', 'Region'],
               axis=1, inplace=True)

    df_01.info()
    merge_01 = pd.merge(df_base, df_01, how='left', on=['Year', 'Month', 'Portfolio Manager ID'])
    merge_01.info()
    #check_null_df(merge_01)
    #merge_01.to_csv(os.getcwd() + '/csv/testmerge1.csv', index=False)

    df_02 = grouped.get_group('Natural Gas')
    df_02.rename(columns={'Usage/Quantity':'gas_amt',
                          'Usage Units':'gas_unit',
                          'Cost ($)':'gas_cost',
                          'Portfolio Manager Meter ID':'gas_meter_id'},
                 inplace=True)
    df_02.drop(['Building ID', 'State', 'Country', 'Postal Code',
                'Year Built', 'GSF', 'Region', 'Meter Type'],
               axis=1, inplace=True)
    merge_02 = pd.merge(merge_01, df_02, how='left', on=['Year', 'Month', 'Portfolio Manager ID'])
    merge_02.info()

    df_03 = grouped.get_group('Fuel Oil (No. 2)')
    df_03.rename(columns={'Usage/Quantity':'oil_amt',
                          'Usage Units':'oil_unit',
                          'Cost ($)':'oil_cost',
                          'Portfolio Manager Meter ID':'oil_meter_id'},
                 inplace=True)
    df_03.drop(['Building ID', 'State', 'Country', 'Postal Code',
                'Year Built', 'GSF', 'Region', 'Meter Type'],
               axis=1, inplace=True)
    merge_03 = pd.merge(merge_02, df_03, how='left', on=['Year', 'Month', 'Portfolio Manager ID'])
    merge_03.info()

    df_04 = grouped.get_group('Potable: Mixed Indoor/Outdoor')
    df_04.rename(columns={'Usage/Quantity':'water_amt',
                          'Usage Units':'water_unit',
                          'Cost ($)':'water_cost',
                          'Portfolio Manager Meter ID':'water_meter_id'},
                 inplace=True)
    df_04.drop(['Building ID', 'State', 'Country', 'Postal Code',
                'Year Built', 'GSF', 'Region', 'Meter Type'],
               axis=1, inplace=True)
    merge_04 = pd.merge(merge_03, df_04, how='left', on=['Year', 'Month', 'Portfolio Manager ID'])
    merge_04.drop(['Meter Type', 'Country'], axis=1, inplace=True)
    merge_04.info()
    output = merge_04.drop_duplicates()
    output.info()
    output.to_csv(os.getcwd() + '/csv/testmerge2.csv', index=False)

    merge_04 = merge_04[merge_04['Building ID'].isin(euas_set)]
    logger.debug('Number of buildings after filter with EUAS membership')
    logger.debug(len(merge_04['Building ID'].unique()))
    merge_04.info()
    merge_04.to_csv(os.getcwd() + '/csv/testmerge-euas.csv', index=False)
    '''

main()
