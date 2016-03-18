import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import pylab as P
import matplotlib.pyplot as plt
from scipy import stats
from scipy import optimize
import pyqt_fit.nonparam_regression as smooth
from pyqt_fit import npr_methods
import calendar
import re
import textwrap as tw
from geopy.geocoders import Nominatim
# from geopy.distance import vincenty
import time
import geocoder
from vincenty import vincenty

homedir = os.getcwd() + '/csv_FY/weather/'
weatherdir = os.getcwd() + '/csv_FY/weather/'

def is_string(s):
    # float, with possible trailing e-0x
    pattern = re.compile('-?[0-9]{0,3}\.?[0-9]{0,20}(e-[0-9]{2})?$')
    if type(s) is str:
        return (not re.match(pattern, s))
    else:
        return False

def check_data_single(f):
    filename = f[f.rfind('/') + 1:]
    s = filename[:4]
    outfile = (weatherdir + 'weather_nostr/{0}'.format(filename))
    df = pd.read_csv(f)
    col_value = df[s].tolist()
    str_value = [x for x in col_value if is_string(x)]
    str_set = set(str_value)
    if len(str_set) > 0:
        print s, list(str_set)
        df.replace(list(str_set), np.nan, inplace=True)
    df.to_csv(outfile, index=False)

def get_mean_temp_single(f):
    filename = f[f.rfind('/') + 1:]
    outfile = (weatherdir + 'station_temp/{0}'.format(filename))
    print 'write mean temperature to {0}'.format(filename)
    df = pd.read_csv(f)
    df.drop('Timestamp', axis=1, inplace=True)
    df.set_index(pd.DatetimeIndex(df['localTime']), inplace=True)
    df = df.resample('M', how = 'mean').to_csv(outfile, index=True)
    df2 = pd.read_csv(outfile)
    df2['year'] = df2['Unnamed: 0'].map(lambda x: x[:4])
    df2['month'] = df2['Unnamed: 0'].map(lambda x: int(x[5: 7]))
    df2.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
    cols = list(df2)
    cols.remove('year')
    cols.remove('month')
    cols.insert(1, 'month')
    cols.insert(1, 'year')
    df2 = df2[cols]
    df2.to_csv(outfile, index=False)

def get_DD_itg_single(f, baselist, theme):
    starttime = time.time()
    df = pd.read_csv(f)
    filename = f[f.rfind('/') + 1:]
    s = filename[:4]
    df.drop('Timestamp', axis=1, inplace=True)
    df.set_index(pd.DatetimeIndex(df['localTime']), inplace=True)
    df_hour = df.resample('H', how = 'mean')

    for base in baselist:
        base_hd = '{0}F'.format(base)
        if theme == 'HDD':
            df_hour[base_hd] = df_hour[s].map(lambda x: 0 if x >= base else base - x)
        else:
            df_hour[base_hd] = df_hour[s].map(lambda x: 0 if x <= base else x - base)
    df_day = df_hour.resample('D', how = 'mean')
    df_month = df_day.resample('M', how = 'sum').to_csv(weatherdir + 'station_dd/{0}_{1}.csv'.format(theme, s), index=True)
    print 'calculating {0} for {1} in {2} s'.format(theme, s, round(time.time() - starttime, 2))

def process_single_stations():
    filelist = glob.glob(weatherdir + 'weatherinput/by_station/*.csv')
    length = len(filelist)
    for i in range(length):
        f = filelist[i]
        print i
        check_data_single(f)

def process_weatherfile():
    # process_single_stations()
    filelist = glob.glob(weatherdir + 'weather_nostr/*.csv')
    length = len(filelist)
    baselist = range(40, 81)
    for i in range(length):
        f = filelist[i]
        print i
        get_mean_temp_single(f)
        get_DD_itg_single(f, baselist, 'HDD')
        get_DD_itg_single(f, baselist, 'CDD')
    return

def main():
    process_weatherfile()
    return

main()
