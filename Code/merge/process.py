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
    euas_set = set(df_t['Building ID'].tolist())
    df_t = df_t.drop_duplicates()
    df_t.info()

    df = pd.read_csv(os.getcwd() + '/csv/cleaned/static_info.csv', usecols = ['Building ID'])
    df = df.drop_duplicates()
    pm_set = set(df['Building ID'].tolist())
    common_id_set = pm_set.intersection(euas_set)
    all_id_set = pm_set.union(euas_set)
    df_all = pd.DataFrame({'Building ID' : pd.Series(list(all_id_set))})
    df_01 = pd.merge(df_all, df_t, on='Building ID', how='left')
    df_01.info()
    df_01['pm'] = df_01['Building ID'].map(lambda x: x if x in pm_set else np.nan)
    df_01['euas'] = df_01['Building ID'].map(lambda x: x if x in euas_set else np.nan)
    df_01.to_csv(os.getcwd() + '/csv/cleaned/euas_pm_cmp.csv', index=False)

main()
