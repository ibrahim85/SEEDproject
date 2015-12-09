import pandas as pd
import os
import glob

## ## ## ## ## ## ## ## ## ## ##
## logging and debugging logger.info settings
import logging
import sys

logger = logging.Logger('reading')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


# return a data frame with key : Portfolio Manager ID
# separate sheets
indir = os.getcwd() + '/csv/'
outdir = os.getcwd() + '/output/'

def read_static():
    csv = indir + 'sheet-0.csv'
    logger.debug('read static info')
    df = pd.read_csv(csv)
    # take the five digits of zip code
    df['Property Name'] = df['Property Name'].map(lambda x: x[:x.find('-')])
    df['Postal Code'] = df['Postal Code'].map(lambda x: x[:5])
    logger.debug(df[:5])

    # read usage
    csv = indir + 'sheet-2.csv'
    logger.debug('read usage info')
    df2 = pd.read_csv(csv)
    df2 = df2.drop('Property Name', 1)
    logger.debug(df2[:5])

    # Need to know which one to keep
    # only return one use type
    df_join = df.join(df2, on = 'Portfolio Manager ID', lsuffix = '_l', rsuffix = '_r', sort = True)
    # keep all use types
    #df_join = df2.join(df, on = 'Portfolio Manager ID', lsuffix = '_l', rsuffix = '_r', sort = True)
    print '{0} rows in df_join'.format(len(df_join.index))
    logger.debug(df_join[:5])

read_static()
