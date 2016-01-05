import pandas as pd
import numpy as np
import glob
import os

df_static = pd.read_csv(os.getcwd() + '/csv/cleaned/static_info.csv',
        usecols = ['Building ID', 'GSF', 'Region', 'Portfolio Manager ID'])

filelist = glob.glob(os.getcwd() + '/csv/single_building_allinfo/' + '*.csv')
for csv in filelist[:1]:
    filename = csv[csv.find('pm'):]
    print '\n\nprocessing ' + filename
    df = pd.read_csv(csv)
    df['Start Date'] = df['Start Date'].map(lambda x: np.datetime64(x[:10], 'D'))
    df['End Date'] = df['End Date'].map(lambda x: np.datetime64(x[:10], 'D'))

    print('Null value count of \'Cost ($)\' before fillna')
    print df['Cost ($)'].isnull().value_counts()
    print('Fill \'Cost ($)\' with -1')
    df['Cost ($)'].fillna(-1, inplace=True)
    print('Null value count of \'Cost ($)\' after fillna')
    print df['Cost ($)'].isnull().value_counts()

    print('Null value count of \'End Date\' before drop null')
    print df['End Date'].isnull().value_counts()
    print('Drop null value of \'End Date\'')
    df.dropna(inplace=True)
    print('Null value count of \'End Date\'after drop null')
    print (df['End Date'].isnull().value_counts())

    df['days'] = (df['End Date'] - df['Start Date']) / np.timedelta64(1, 'D')
    df['ave'] = df['Usage/Quantity']/df['days']
    df = df.set_index('Start Date')
    df.drop(['Portfolio Manager ID', 'Portfolio Manager Meter ID', 'Usage/Quantity', 'End Date', 'Cost ($)', 'days'], axis=1, inplace=True)
    grouped = df.groupby('Meter Type')
    print grouped.size()

    for name, group in grouped:
        print '\n------------------------------------------'
        #if name == 'Electric - Solar':
        print name
        group = group.resample('D', fill_method = 'ffill')
        group.to_csv(os.getcwd() + 'csv/single-building-sub/')
