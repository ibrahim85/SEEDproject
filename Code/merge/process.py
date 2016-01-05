import pandas as pd
import numpy as np
import os
import glob

## ## ## ## ## ## ## ## ## ## ##
## logging and debugging logger.info settings
import logging
import sys

logger = logging.Logger('reading')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def main():
    euas = os.getcwd() + '/input/EUAS.csv'
    logger.debug('Read in EUAS region')
    df_t = pd.read_csv(euas, usecols=['Building ID'])
    df_t['Building ID'] = df_t['Building ID'].map(lambda x: x.partition(' ')[0][:8])
    euas_set = set(df_t['Building ID'].tolist())
    df_t = df_t.drop_duplicates()
    df_t.info()

    df = pd.read_csv(os.getcwd() + '/csv/cleaned/static_info.csv', usecols = ['Building ID'])
    df = df.drop_duplicates()
    pm_set = set(df['Building ID'].tolist())
    common_id_set = pm_set.intersection(euas_set)
    all_id_set = pm_set.union(euas_set)
    print
    print 'total records {0}'.format(len(all_id_set))
    print 'matching records {0}'.format(len(common_id_set))
    df_all = pd.DataFrame({'Building ID' : pd.Series(list(all_id_set))})
    df_01 = pd.merge(df_all, df_t, on='Building ID', how='left')
    df_01.info()
    df_01['pm'] = df_01['Building ID'].map(lambda x: x if x in pm_set else np.nan)
    df_01['euas'] = df_01['Building ID'].map(lambda x: x if x in euas_set else np.nan)
    df_01.to_csv(os.getcwd() + '/csv/cleaned/euas_pm_cmp.csv', index=False)

    logger.debug('original buildings')
    df_energy = pd.read_csv(os.getcwd() + '/csv/select_column/sheet-5.csv')
    df_energy.rename(columns={'Portfolio Manager ID':'Building ID'}, inplace=True)
    grouped = df_energy.groupby('Meter Type')
    b1 = set(grouped.get_group('Electric - Grid')['Building ID'].tolist())
    b2 = set(grouped.get_group('Natural Gas')['Building ID'].tolist())
    b3 = set(grouped.get_group('Fuel Oil (No. 2)')['Building ID'].tolist())
    b4 = set(grouped.get_group('Potable: Mixed Indoor/Outdoor')['Building ID'].tolist())
    logger.debug('number of buildings with the four major energy source')
    logger.debug(len(b1.union(b2.union(b3.union(b4)))))

    logger.debug('\nafter cleaned')
    df_energy = pd.read_csv(os.getcwd() + '/csv/cleaned/energy_info.csv')
    df_energy.rename(columns={'Portfolio Manager ID':'Building ID'}, inplace=True)
    grouped = df_energy.groupby('Meter Type')
    b1 = set(grouped.get_group('Electric - Grid')['Building ID'].tolist())
    b2 = set(grouped.get_group('Natural Gas')['Building ID'].tolist())
    b3 = set(grouped.get_group('Fuel Oil (No. 2)')['Building ID'].tolist())
    b4 = set(grouped.get_group('Potable: Mixed Indoor/Outdoor')['Building ID'].tolist())
    logger.debug('number of buildings with the four major energy source')
    common = (b1.union(b2.union(b3.union(b4))))
    logger.debug(len(common))

main()
