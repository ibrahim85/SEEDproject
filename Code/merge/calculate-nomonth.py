import pandas as pd
import numpy as np
import os
import glob

# question:
# 1. for not completed year, how to calculate EUI

## ## ## ## ## ## ## ## ## ## ##
## logging and debugging logger.info settings
import logging
import sys

logger = logging.Logger('reading')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def check_unit_type(df):
    grouped = df.groupby('Meter Type')
    for name, group in grouped:
        if name == 'Natural Gas':
            logger.debug('Gas consumption unit types')
            logger.debug(group.groupby('Usage Units').size())
            df['Usage Units'].fillna('', inplace=True)
        elif name == 'Electric - Grid':
            logger.debug('Electric - Grid consumption unit types')
            logger.debug(group.groupby('Usage Units').size())

def check_unit_convert(df):
    grouped_unit = df.groupby(['Meter Type', 'Usage Units'])
    for name, group in grouped_unit:
        if name[0] == 'Natural Gas':
            # iloc: integer position based (from 0 to length-1 of the axis)
            # because the index of the entries in the group is the same as original
            print group.iloc[:10, [11, 12, 15]]

def check_total_amt(df):
    grouped_unit = df.groupby('gas_unit')
    for name, group in grouped_unit:
        print name
        print group.iloc[:10, [10, 11, 14, 15, 25, 26]]

def calculate():
    # read customized table
    df = pd.read_csv(os.getcwd() + '/csv/originalShape.csv')
    logger.debug('reading file testmerge-euas.csv')
    # no need to fillna for empty measurement because summing will be zero

    # multiplier for gas gasconsumption, unit conversion to kbtu
    gas_multiply = {'cf (cubic feet)' : 1.026, 'therms' : 100, '' : 0}
    #df['gas_amt_kbtu'] = df.apply(lambda row: 1 if row['Meter Type'] != 'Natural Gas' else row['Usage/Quantity'] * gas_multiply[row['Usage Units']], axis=1)
    df['Usage/Quantity'] = df.apply(lambda row: 1 if row['Meter Type'] != 'Natural Gas' else row['Usage/Quantity'] * gas_multiply[row['Usage Units']], axis=1)
    df.info()

    grouped = df.groupby(['Year', 'Building ID', 'Meter Type'])
    id_list = []
    region_list = []
    year_list=[]
    type_list=[]
    eui_list = []
    for name, group in grouped:
        id_list.append(group.iloc[0, 0])
        region_list.append(group.iloc[0, 6])
        year_list.append(group.iloc[0, 15])
        type_list.append(group.iloc[0, 9])
        eui_list.append(group['Usage/Quantity'].sum()/group['GSF'].mean())
    df_eui = pd.DataFrame({'Building ID':id_list, 'Region':region_list,
        'Year':year_list, 'EUI':eui_list, 'Meter Type':type_list})
    df_eui.info()

    logger.debug('Nan EUI counts:')
    logger.debug(df_eui['EUI'].isnull().value_counts())
    df_eui['EUI'].replace(np.inf, np.nan, inplace=True)
    logger.debug(df_eui['EUI'].isnull().value_counts())
    df_eui.dropna(inplace=True)
    logger.debug(df_eui['EUI'].isnull().value_counts())
    print df_eui['EUI'].value_counts()
    ck.get_range(df_eui)
    # BOOKMARK: ADD NATURAL GAS AND ELECTRICITY

    df_eui.to_csv(os.getcwd() + '/csv/eui-nomonth.csv', index=False)

import matplotlib.pyplot as plt
import util_check as ck
import pylab as P
import seaborn as sns

def plot():
    # read eui table
    df = pd.read_csv(os.getcwd() + '/csv/eui-nomonth.csv')
    logger.debug('finished reading file eui.csv')
    grouped = df.groupby('Meter Type')
    gas = grouped.get_group('Natural Gas')
    elec = grouped.get_group('Electric - Grid')
    # bookmark

    '''
    df.boxplot(column='EUI', by='Region')
    plt.ylabel('EUI')
    plt.xlabel('Region')
    plt.title('EUI by Region')
    P.savefig(os.getcwd() + '/plot/EUIbyRegion.png')
    plt.close()

    grouped = df.groupby('Region')
    for name, group in grouped:
        sns.distplot(group['EUI'])
        plt.xlabel('EUI')
        plt.title('EUI Distribution')
        P.savefig(os.getcwd() + '/plot/Region-' + str(name) + '-EUIdistribution.png')
        plt.close()
    df = df[df['Region'] != 3]
    df = df[df['Region'] != 11]
    df.sort(columns='Region', inplace=True)
    sns.violinplot(x = 'Region', y = 'EUI', data = df)
    plt.show()
    '''

def main():
    #calculate()
    plot()

main()
