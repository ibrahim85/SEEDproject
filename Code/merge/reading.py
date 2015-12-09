# separate excel sheets to csv files
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
indir = os.getcwd() + '/input/'
outdir = os.getcwd() + '/csv/'

def excel2csv_single(excel, i, col_list):
    df = pd.read_excel(excel, sheetname=int(i), skiprows=4, header=5,
                       parse_cols = col_list)
    file_out = outdir + 'sheet-{0}'.format(i) + '.csv'
    df.to_csv(file_out, index=False)

def excel2csv():
    filelist = glob.glob(indir + '*.xlsx')
    logger.debug('files to read {0}'.format(filelist))
    col_dict = {'0':[0, 1, 5, 7, 9, 12], '2':[0, 1, 3], '5':[1, 2, 4, 6, 8, 10]}
    for excel in filelist:
        for i in ['0', '2', '5']:
            print 'reading sheet {0}'.format(i)
            excel2csv_single(excel, i, col_dict[i])

excel2csv()
