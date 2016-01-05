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
    logger.debug('Gas consumption unit types')
    logger.debug(df.groupby('gas_unit').size())
    logger.debug('Electric - Grid consumption unit types')
    logger.debug(df.groupby('elec_unit').size())

def check_unit_convert(df):
    grouped_unit = df.groupby('gas_unit')
    for name, group in grouped_unit:
        print name
        # iloc: integer position based (from 0 to length-1 of the axis)
        # because the index of the entries in the group is the same as original
        print group.iloc[:10, [14, 15, 25]]

def check_total_amt(df):
    grouped_unit = df.groupby('gas_unit')
    for name, group in grouped_unit:
        print name
        print group.iloc[:10, [10, 11, 14, 15, 25, 26]]

def calculate():
    # read customized table
    df = pd.read_csv(os.getcwd() + '/csv/testmerge-euas.csv')
    logger.debug('reading file testmerge-euas.csv')
    # no need to fillna for empty measurement because summing will be zero

    # check unit type of gas and electricity
    check_unit_type(df)
    # multiplier to kbtu
    df['gas_unit'].fillna('', inplace=True)

    # multiplier for gas gasconsumption, unit conversion to kbtu
    gas_multiply = {'cf (cubic feet)' : 1.026, 'therms' : 100, '' : 0}
    df['gas_amt_kbtu'] = df.apply(lambda row: row['gas_amt'] * gas_multiply[row['gas_unit']], axis=1)

    # checking converted result
    #check_unit_convert(df)

    df['total_amt'] = df['elec_amt'] + df['gas_amt_kbtu']
    # checking added energy consumption
    #check_total_amt(df)

    grouped = df.groupby(['Year', 'Building ID'])
    id_list = []
    region_list = []
    year_list=[]
    eui_list = []
    for name, group in grouped:
        id_list.append(group.iloc[0, 0])
        region_list.append(group.iloc[0, 5])
        year_list.append(group.iloc[0, 7])
        eui_list.append(group['total_amt'].sum()/group['GSF'].mean())
    df_eui = pd.DataFrame({'Building ID':id_list, 'Region':region_list,
                           'Year':year_list, 'EUI':eui_list})
    df_eui.info()
    logger.debug('Nan EUI counts:')
    logger.debug(df_eui['EUI'].isnull().value_counts())
    df_eui['EUI'].replace(np.inf, np.nan, inplace=True)
    logger.debug(df_eui['EUI'].isnull().value_counts())
    df_eui.dropna(inplace=True)
    logger.debug(df_eui['EUI'].isnull().value_counts())
    print df_eui['EUI'].value_counts()
    ck.get_range(df_eui)
    df_eui.to_csv(os.getcwd() + '/csv/eui.csv', index=False)

import matplotlib.pyplot as plt
import util_check as ck
import pylab as P

def plot():
    # read eui table
    df = pd.read_csv(os.getcwd() + '/csv/eui.csv')
    logger.debug('finished reading file eui.csv')

    #df = df[df['Region'] < 11]
    df.boxplot(column='EUI', by='Region')
    plt.ylabel('EUI')
    plt.xlabel('Region')
    plt.title('EUI by Region')
    P.savefig(os.getcwd() + '/plot2/EUIbyRegion.png')
    plt.close()

    import seaborn as sns
    grouped = df.groupby('Region')
    for name, group in grouped:
        sns.distplot(group['EUI'])
        plt.xlabel('EUI')
        plt.title('EUI Distribution')
        P.savefig(os.getcwd() + '/plot2/Region-' + str(name) + '-EUIdistribution.png')
        plt.close()
    df.sort(columns='Region', inplace=True)
    sns.violinplot(x = 'Region', y = 'EUI', data = df)
    plt.ylabel('EUI')
    plt.xlabel('Region')
    plt.title('EUI by Region Violin Plot')
    P.savefig(os.getcwd() + '/plot2/EUIbyRegionViolin.png')
    plt.close()

def main():
    calculate()
    plot()

main()
