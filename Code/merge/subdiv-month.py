import pandas as pd
import numpy as np
import glob
import os

def fillna_verbo(df, col, value, isverb):
    if isverb:
        print('Null value count of \'{0}\' before fillna'.format(col))
        print df[col].isnull().value_counts()
        print('Fill \'{0}\' with {1}'.format(col, value))
    df[col].fillna(value, inplace=True)
    if isverb:
        print('Null value count of \'{0}\' after fillna'.format(col))
        print df[col].isnull().value_counts()

def dropna_verbo(df, col, isverb):
    if isverb:
        print('Null value count of \'{0}\' before drop null'.format(col))
        print df[col].isnull().value_counts()
        print('Drop null value of \'{0}\''.format(col))
    df.dropna(inplace=True)
    if isverb:
        print('Null value count of \'{0}\'after drop null'.format(col))
        print (df[col].isnull().value_counts())

def unit_convert(quantity, meter_type, unit):
    if meter_type == 'Natural Gas':
        return quantity * gas_multiply[unit]
    elif meter_type == 'Electric - Grid':
        return quantity * ele_multiply[unit]
    else:
        return quantity

filelist = glob.glob(os.getcwd() + '/csv/single_building/' + '*.csv')
# print filelist.index(os.getcwd() + '/csv/single_building/' + 'pm-20600.csv')
# 303
for csv in filelist:
    filename = csv[csv.find('pm'):]
    print '\n\nprocessing ' + filename
    df = pd.read_csv(csv)
    # filter out empty start end date
    df = df[df['Start Date'] != '0']
    df = df[df['End Date'] != '0']
    df['Start Date'] = df['Start Date'].map(lambda x: np.datetime64(x[:10], 'D'))
    df['End Date'] = df['End Date'].map(lambda x: np.datetime64(x[:10], 'D'))

    fillna_verbo(df, 'Cost ($)', 0, False)
    fillna_verbo(df, 'Usage Units', '', False)
    dropna_verbo(df, 'End Date', False)

    # unit conversion
    gas_multiply = {'cf (cubic feet)' : 1.026, 'therms' : 100,
                    'GJ': 947.817, '' : 0}
    ele_multiply = {'kWh (thousand Watt-hours)' : 3.412,
                    'GJ': 947.817, '' : 0}
    df['Usage/Quantity'] = df.apply(lambda row: unit_convert(row['Usage/Quantity'], row['Meter Type'], row['Usage Units']), axis=1)

    # get interval
    df['days'] = (df['End Date'] - df['Start Date']) / np.timedelta64(1, 'D')
    # get average consumption per day
    df = df.set_index(pd.DatetimeIndex(df['End Date']))
    df.drop(['End Date', 'days'],
            axis=1, inplace=True)
    grouped = df.groupby('Meter Type')
    #print grouped.size()

    # upsample data
    outdir = os.getcwd() + '/csv/single_building_sub/'
    def get_subdir(name):
        if name in ['Electric - Grid', 'Natural Gas', 'Fuel Oil (No. 2)']:
            return name + '/'
        elif name == 'Potable: Mixed Indoor/Outdoor':
            return 'Water/'
        else:
            return 'Other/'
    for name, group in grouped:
        #print '\n------------------------------------------'
        #print name
        if name == 'Natural Gas' or name == 'Electric - Grid':
            group['Usage Units'] = 'kBtu'
        group_up = group.resample('M', how = 'sum', fill_method = None)
        group_up = group_up.dropna()
        outfilename = filename[:-4] + '-' + str(name) + '.csv'
        outfilename = outfilename.replace('/', 'or')
        outfilename = outfilename.replace(':', '-')
        #print outfilename
        group_up['Meter Type'] = name
        group_up['Usage Units'] = group['Usage Units'][0]
        group_up.to_csv(outdir + get_subdir(name) + outfilename,
                        index=True, index_label='End Date')
