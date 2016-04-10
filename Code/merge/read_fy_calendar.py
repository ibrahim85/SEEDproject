import pandas as pd
import os
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as P
from geopy.geocoders import Nominatim
from vincenty import vincenty
import time
import geocoder
import requests
import textwrap as tw
import sqlite3
# from geopy.distance import vincenty
# 'Cat' appear in file FY14, FY14, not FY13, this version account for this

homedir = os.getcwd() + '/csv_FY/'
weatherdir = os.getcwd() + '/csv_FY/weather/'

def check_use_dupe():
    df = pd.read_csv(os.getcwd() + '/csv/all_column/sheet-0-all_col.csv')
    df = df[['Property Name', 'Self-Selected Primary Function']]
    df['Property Name'] = df['Property Name'].map(lambda x: x.partition(' ')[0][:8])
    df['dup'] = df.duplicated(cols = 'Property Name')
    df.info()
    df.to_csv(os.getcwd() + '/csv/all_column/sheet-0-dupuse.csv', index=False)

def get_office():
    filename = os.getcwd() + '/csv/all_column/sheet-0-all_col.csv'
    df = pd.read_csv(filename)
    df = df[['Property Name', 'Self-Selected Primary Function']]
    df['Property Name'] = df['Property Name'].map(lambda x: x.partition(' ')[0][:8])
    df = df[df['Self-Selected Primary Function'] == 'Office']
    df.drop_duplicates(cols='Property Name', inplace=True)
    print len(df)
    return set(df['Property Name'].tolist())

# get a set of buildings of a dataframe
def get_building(df):
    return set(df['Building Number'].tolist())

# concatnate fiscal year data to one file a year, only for fiscal year,
# becuase there is different columns
def get_raw_concat(yearlist, calOrFiscal):
    print 'concatnate {0} year data to one file a year'.format(calOrFiscal)
    if calOrFiscal == 'fis':
        conn = sqlite3.connect(homedir + \
                               'db/EUAS_input_by_fiscal_year.db')
        year_col = 'Fiscal Year'
        month_col = 'Fiscal Month'
        pre = 'FY'
        folder = 'sep'
    else:
        conn = sqlite3.connect(homedir + \
                               'db/EUAS_input_by_calendar_year.db')
        year_col = 'year'
        month_col = 'month'
        pre = 'CY'
        folder = 'cal'
    c = conn.cursor()

    for year in yearlist:
        print '    concatenating {1}{0}'.format(year, pre)
        label = str(int(year))[-2:]
        filelist = ['{0}/csv_FY/{3}/{4}{1}_{2}.csv'.format(os.getcwd(), label, i, folder, pre) for i in range(1, 12)]
        dfs = [pd.read_csv(f) for f in filelist]
        # for i, df in enumerate(dfs):
        #     if 'AK0000AA' in df['Building Number'].tolist():
        #         print filelist[i]
        df_all = pd.concat(dfs, join='inner', ignore_index=True)
        df_all.sort(columns=['Building Number', month_col], inplace=True)
        df_all.to_csv(os.getcwd() + \
                '/csv_FY/raw_concat/{0}{1}.csv'.format(pre, label),
                index=False)
        df_all.to_sql('{0}{1}'.format(pre, label), conn,
                      if_exists='replace')
    conn.close()

def get_state_abbr_dict():
    df = pd.read_csv(os.getcwd() + '/input/FY/state2abbr.csv')
    return dict(zip(df['Postal'], df['State']))

import operator
import itertools
def get_most_common(L):
    if len(L) == 0:
        return None
    SL = sorted([(x, i) for (i, x) in enumerate(L)],
                key=operator.itemgetter(0))
    groups = itertools.groupby(SL, key=operator.itemgetter(0))
    group_len_list = [(k, len(list(g))) for k, g in groups]
    return sorted(group_len_list, key=operator.itemgetter(1),
                  reverse=True)[0][0]

def sanity_check_static_long(yearlist):
    print '    checking static info ...'
    labellist = [str(int(yr))[-2:] for yr in yearlist]
    dfs = []
    check_cols = ['Region No.', 'State', 'Cat', 'Gross Sq.Ft']
    cols = ['Building Number', 'Fiscal Year'] + check_cols
    for yr in yearlist:
        label = str(int(yr))[-2:]
        df = pd.read_csv(os.getcwd() + '/csv_FY/raw_concat/FY{0}_drop.csv'.format(label))
        if not 'Cat' in df:
            df['Cat'] = np.nan
        df = df[cols]
        df.dropna(subset=['Building Number'], inplace=True)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.replace(get_state_abbr_dict(), inplace=True)
    df_all.replace('******', np.nan, inplace=True)
    df_all.sort(columns=['Region No.', 'Building Number', 'Fiscal Year'], inplace=True)
    df_all.to_csv(homedir + 'master_table/static_info_EUAS_long.csv',
                  index=False)
    df = df_all.drop('Fiscal Year', axis=1)
    dfs = []
    for col in check_cols:
        colname = 'Unique value count of {0}'.format(col)
        df_temp = (df.groupby('Building Number')[col].nunique()).to_frame(name=colname)
        df_temp = df_temp[df_temp[colname] > 1]
        if len(df_temp) > 0:
            df_temp.to_csv(homedir + \
                           'master_table/EUAS_static_{0}_unique_count.csv'.format(col))
    df_tidy = df.groupby('Building Number', as_index=False).last()
    df_tidy.drop('Gross Sq.Ft', axis=1, inplace=True)
    df_tidy.sort(columns=['Region No.', 'Building Number'],
                 inplace=True)
    df_tidy.to_csv(homedir + \
                   'master_table/EUAS_static_tidy.csv'.format(col),
                   index=False)
    conn = sqlite3.connect(homedir + 'db/static_info_tidy.db')
    c = conn.cursor()
    df_tidy.to_sql('EUAS_static_tidy', conn, if_exists='replace')
    return

def sanity_check_static(yearlist):
    conn = sqlite3.connect(homedir + 'db/static_info.db')
    c = conn.cursor()
    print '    checking static info ...'
    labellist = [str(int(yr))[-2:] for yr in yearlist]
    dfs = []
    rename_cols = ['Region No.', 'State', 'Cat', 'Gross Sq.Ft']
    static_cols = rename_cols + ['Building Number']
    def suf(col, label):
        return '{0}_{1}'.format(col, label)
    def is_changing(r, cols):
        values = [r[col] for col in cols]
        values = list(set(values))
        values = [v for v in values if not (type(v) == float and
                                            np.isnan(v))]
        return len(values) > 1

    for yr in yearlist:
        label = str(int(yr))[-2:]
        df = pd.read_csv(os.getcwd() + '/csv_FY/raw_concat/FY{0}_drop.csv'.format(label))
        if not 'Cat' in df:
            df['Cat'] = np.nan
        df = df[static_cols]
        rename_dict = dict(zip(rename_cols, [suf(x, label) for x in
                                             rename_cols]))
        df.dropna(subset=['Building Number'], inplace=True)
        df.rename(columns=rename_dict, inplace=True)
        dfs.append(df)
    df_all = reduce(lambda x, y: pd.merge(x, y, how='outer', on='Building Number'), dfs)
    df_all.replace(get_state_abbr_dict(), inplace=True)
    df_all.replace('******', np.nan, inplace=True)
    changing_summary = []
    for col in rename_cols:
        cols = [c for c in list(df_all) if col in c]
        c = '{0}_is_changing'.format(col)
        df_all[c] = df_all.apply(lambda r: is_changing(r, cols),
                                 axis=1)
        changing_summary.append(c)
    newcols = ['Building Number'] + changing_summary + \
        reduce(lambda x, y: x + y, [[suf(c, l) for l in labellist] for
                                    c in rename_cols])
    df_all = df_all[newcols]
    df_all.sort(columns=['Building Number'], inplace=True)
    df_all.to_csv(os.getcwd() + '/csv_FY/master_table/static_info_EUAS_check.csv', index=False)
    df_all.to_sql('static_info_EUAS_check', conn, if_exists='replace')
    def get_value(r, cols):
        values = [r[col] for col in cols]
        values = list(set(values))
        values = [v for v in values if not (type(v) == float and
                                            np.isnan(v))]
        if len(values) == 0:
            return np.nan
        else:
            return values[0]

    def get_latest(r, cols):
        values = [r[col] for col in cols]
        values = list(set(values))
        values = [v for v in values if not (type(v) == float and
                                            np.isnan(v))]
        if len(values) == 0:
            return np.nan
        else:
            return values[-1]

    def get_mode(r, cols):
        values = [r[col] for col in cols]
        values = list(values)
        values = [v for v in values if not (type(v) == float and
                                            np.isnan(v))]
        return get_most_common(values)

    drop_cols = []
    df_all.drop(['Region No._is_changing', 'State_is_changing'],
                axis=1, inplace=True)
    for col in ['Region No.', 'State']:
        cols = [c for c in list(df_all) if col in c]
        df_all[col] = df_all.apply(lambda r: get_value(r, cols),
                                   axis=1)
        drop_cols += cols
    cols = [c for c in list(df_all) if 'Cat' in c]
    cols.remove('Cat_is_changing')
    df_all['Cat_latest'] = df_all.apply(lambda r: get_latest(r, cols),
                                        axis=1)
    df_all['Cat_most_common'] = df_all.apply(lambda r:
                                             get_mode(r, cols), axis=1)
    df_all['Cat_latest_eq_mode'] = df_all.apply(lambda r:
                                                r['Cat_latest'] == \
                                                r['Cat_most_common'],
                                                axis=1)
    df_all.drop(drop_cols, axis=1, inplace=True)
    df_all.to_csv(os.getcwd() + '/csv_FY/master_table/static_info_from_EUAS.csv', index=False)
    df_all.to_sql('static_info_from_EUAS', conn, if_exists='replace')
    conn.close()
    return

def add_filter_bit(yearlist, calOrFiscal):
    office_set = get_office()
    if calOrFiscal == 'fis':
        pre = 'FY'
        infolder = 'agg'
        yearcol = 'Fiscal Year'
    else:
        pre = 'CY'
        infolder = 'agg_cal'
        yearcol = 'year'
    for yr in yearlist:
        label = str(int(yr))[-2:]
        df_zero = pd.read_csv(os.getcwd() + '/csv_FY/raw_concat/{1}{0}.csv'.format(label, pre))
        df_zero.drop_duplicates(cols='Building Number', inplace=True)
        df_zero.to_csv(os.getcwd() + '/csv_FY/raw_concat/{1}{0}_drop.csv'.format(label, pre), index=False)
        df_zero = df_zero[['Building Number']]
        df = pd.read_csv(os.getcwd() + '/csv_FY/{1}/eui_{0}.csv'.format(yr, infolder))
        def suf(title):
            return '{0}_{1}'.format(title, label)
        df[suf('good_elec')] = df['eui_elec'].map(lambda x: 1 if x >=
                                                  12 else 0)
        df[suf('good_gas')] = df['eui_gas'].map(lambda x: 1 if x >= 3
                                                else 0)
        df[suf('good_water')] = df['eui_water'].map(lambda x: 1 if x
                                                    >= 5 else 0)
        df[suf('good_both')] = df.apply(lambda row: 1 if
                                        row[suf('good_elec')] +
                                        row[suf('good_gas')] == 2 else
                                        0, axis=1)
        df[suf('good_all')] = df.apply(lambda row: 1 if
                                       row[suf('good_elec')] +
                                       row[suf('good_gas')] +
                                       row[suf('good_water')] == 3
                                       else 0, axis=1)
        df_all = pd.merge(df_zero, df, on='Building Number', how = 'outer')
        df_all[suf('good_area')] = df_all['eui'].notnull().map(lambda x: 1 if x else 0)
        df_all[suf('office')] = df_all['Building Number'].map(lambda x: 1 if x in office_set else 0)
        df_all[suf('has_data')] = 1
        df_all.fillna(dict(zip([suf('good_elec'), suf('good_gas'),
                                suf('good_water')], [0, 0, 0])),
                      inplace=True)
        df_all.dropna(subset=['Building Number'], axis=0, inplace=True)
        df_all.to_csv(os.getcwd() +
                      '/csv_FY/filter_bit/{1}/eui_all_20{0}.csv'.format(label, calOrFiscal), index=False)
        df_all.drop(['eui_elec', 'eui_gas', 'eui_oil', 'eui_water', 'eui', 'Region No.', yearcol, 'Cat', 'eui_steam'], axis = 1, inplace=True)
        df_all.to_csv(os.getcwd() +
                      '/csv_FY/filter_bit/{1}/eui_clean_20{0}.csv'.format(label, calOrFiscal), index=False)
        df.to_csv(os.getcwd() +
                  '/csv_FY/filter_bit/{1}/eui_20{0}.csv'.format(label,
                                                                calOrFiscal), index=False)

# merge quality indicator files
def merge_indicator(calOrFiscal):
    print 'start merge_indicator'
    filelist = glob.glob(os.getcwd() + '/csv_FY/filter_bit/{0}/eui_clean_*'.format(calOrFiscal))
    dfs = [pd.read_csv(csv) for csv in filelist]
    all_building = reduce(set.union, [get_building(df) for df in dfs])
    df_base = pd.DataFrame({'Building Number': list(all_building)})
    df_list = [df_base] + dfs
    df_all = reduce(lambda x, y: pd.merge(x, y, how='left', on='Building Number'), df_list)
    df_all.to_csv(os.getcwd() + '/csv_FY/filter_bit/{0}/indicator.csv'.format(calOrFiscal), index=False)
    return

def get_num_char(char, string):
    return len([c for c in string if c == char])

theme_dict = {'has_data': 'has energy data', 
                'good_elec': 'with Electric EUI >= 12', 
                'good_gas': 'with Gas EUI >= 3', 
                'good_both': 'with Electric EUI >= 12 and Gas EUI >= 3'}
# nyear: the number of years of data we want
def plot_building_data_repl(yearlist, theme, calOrFiscal, catfilter,
                            nyear):
    print 'plotting representation for {0}'.format(theme)
    df_temp = pd.read_csv(os.getcwd() + '/csv_FY/filter_bit/{0}/sep_summary/{1}.csv'.format(calOrFiscal, theme))
    if catfilter == 'AI':
        df_temp = df_temp[df_temp['Cat_latest'].isin(['A', 'I'])]
    num_involved = len(set(df_temp['Building Number'].tolist()))
    sumcols = ['{0}_{1}'.format(theme, str(int(yr))[-2:]) \
                for yr in yearlist]
    df_temp = df_temp[sumcols]
    df_temp.rename(columns=lambda x:
                    '20{0}'.format(x[:x.rfind('_') + 1]))
    df_temp.fillna(0, inplace=True)
    df_temp['repl'] = df_temp.apply(lambda r: reduce(lambda x, y: x + y, [str(int(r[c])) for c in list(df_temp)]), axis=1) 
    df_temp = df_temp[['repl']]
    dfs = []
    n_year_data_groups = []
    for name, group in df_temp.groupby('repl'):
        # for graph plotting
        if get_num_char('1', name) >= nyear:
            group['representation'] = name
            dfs.append(group)
            n_year_data_groups.append(name)
    group_dict_nyear = dict(zip(n_year_data_groups, dfs))
    len_values = [len(x) for x in (group_dict_nyear.values())]
    total_bd = reduce(lambda x, y: x + y, len_values)
    group_dict_nyear_dropsmall = {k: group_dict_nyear[k] for k in
                                  group_dict_nyear if
                                  len(group_dict_nyear[k]) > 5}
    order = sorted(group_dict_nyear_dropsmall.keys(), key=lambda x:
                   get_num_char('0', x))
    df_all = pd.concat(group_dict_nyear_dropsmall.values(), ignore_index=True)
    # sns.set_style("white")
    sns.set_context("talk", font_scale=1.5)
    sns.countplot(y='representation', data=df_all, order=order)
    plt.xlabel('Building Count')
    plt.ylabel('building data availability representation')
    plt.title('EUAS data availability {0} to {1}'.format(yearlist[0], yearlist[-1]))
    if catfilter == 'AI':
        plt.suptitle('{0} building (category A/I) in EUAS data set, {1} building {2} for at least {3} year'.format(num_involved, total_bd, theme_dict[theme], nyear))
    else:
        plt.suptitle('{0} building in EUAS data set, {1} building {2} for at least {3} year'.format(num_involved, total_bd, theme_dict[theme], nyear))
    my_dpi = 300
    P.savefig(os.getcwd() + '/plot_FY_annual/EUAS_{0}_{1}.png'.format(theme, catfilter), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi), bbox_inches='tight')
    plt.close()

def sep_indicator(cutoff):
    print 'separating indicator_all.csv into single indicators ...'
    df = pd.read_csv(homedir + 'filter_bit/fis/indicator_all.csv')
    themes = ['has_data', 'good_area', 'good_elec', 'good_gas', 'good_water', 'good_both', 'good_all']
    for theme in themes[:1]:
        df_temp = df.copy()
        cols = ['Building Number', 'Cat_latest', 'office'] + [x for x in list(df_temp) if theme in x]
        df_temp = df_temp[cols]
        df_temp[theme + ' >= {0} years'.format(cutoff)] = \
            df_temp[theme + '_sum'].map(lambda x: 1 if x >= cutoff
                                        else 0) 
        df_temp.to_csv(homedir + \
                       'filter_bit/fis/sep_summary/{0}.csv'.format(theme),
                       index=False)
        df_temp = df_temp[df_temp['Cat_latest'].isin(['A', 'I'])]
        df_temp.to_csv(homedir + \
                       'filter_bit/fis/sep_summary/{0}_AI.csv'.format(theme),
                       index=False)

# report number of each filter, use 'fis'
def report_number(yearlist, calOrFiscal, cutoff):
    def suf(title, yr):
        return '{0}_{1}'.format(title, str(int(yr))[-2:])
    # for separate year:
    print 'report all building'
    df = pd.read_csv(os.getcwd() + '/csv_FY/filter_bit/{0}/indicator.csv'.format(calOrFiscal))

    for yr in yearlist:
        label = str(int(yr))[-2:]
        themes = [suf(x, yr) for x in ['has_data', 'good_area', 'good_elec', 'good_gas', 'good_water', 'good_both', 'good_all']]
        # for theme in themes:
            # print theme
            # print df[theme].sum()
        # print 'office'
        for theme in themes:
            # print 'office: {0}'.format(theme)
            df_temp = df[df[suf('office', yr)] == 1]
            # print df_temp[theme].sum()
    themes = ['has_data', 'good_area', 'good_elec', 'good_gas', 'good_water', 'good_both', 'good_all']

    for theme in themes:
        if theme == 'has_data' or theme == 'good_are':
            sumcols = ['{0}_{1}'.format(theme, str(int(yr))[-2:]) \
                       for yr in yearlist]
            df['{0}_sum'.format(theme)] = df[sumcols].sum(axis=1)
            total_years = len(yearlist)
        else:
            sumcols = ['{0}_{1}'.format(theme, str(int(yr))[-2:]) \
                       for yr in yearlist]
            sumcols.remove(theme + '_16')
            df['{0}_sum'.format(theme)] = df[sumcols].sum(axis=1)
            total_years = len(yearlist) - 1
        df[theme] = df['{0}_sum'.format(theme)].map(lambda x: 1 if x == total_years else 0)
        df[theme + ' >= {0} years'.format(cutoff)] = df['{0}_sum'.format(theme)].map(lambda x: 1 if x >= cutoff else 0)
        # print theme
        # print df[theme].sum()
    
    office_list = get_office()
    df['office'] = df['Building Number'].map(lambda x: 1 if x in office_list else 0)
    # print 'office'
    df_cat = pd.read_csv(homedir + 'master_table/static_info.csv')
    df_cat = df_cat[['Building Number', 'Cat_latest']]
    df_all = pd.merge(df, df_cat, on='Building Number', how='left')
    df_all.to_csv(os.getcwd() + '/csv_FY/filter_bit/{0}/indicator_all.csv'.format(calOrFiscal), index=False)
    df_all = df_all[df_all['Cat_latest'].isin(['A', 'I'])]
    df_all.to_csv(os.getcwd() + '/csv_FY/filter_bit/{0}/indicator_all_catAI.csv'.format(calOrFiscal), index=False)
    df_temp = df_all[df_all['office'] == 1]
    # for theme in themes:
        # print 'office: {0}'.format(theme)
        # print df_temp[theme].sum()

# process starts from positive floor area
def get_flow_reorg(yearlist, calOrFiscal, cutoff):
    add_filter_bit(yearlist, calOrFiscal)
    merge_indicator(calOrFiscal)
    report_number(yearlist, calOrFiscal, cutoff)
    sep_indicator(cutoff)
    return

def check_num_bd(dfs):
    buildings = [get_building(df) for df in dfs]
    return [len(b) for b in buildings]

# check number of common buildings of two list of data frames
def check_common_bd_pair(dfs_1, dfs_2):
    buildings_1 = [get_building(df) for df in dfs_1]
    buildings_2 = [get_building(df) for df in dfs_2]
    assert(len(buildings_1) == len(buildings_2))
    return [len(buildings_1[i].intersection(buildings_2[i]))
            for i in range(len(buildings_1))]

def check_sheetname(excel, flag):
    if flag:
        excelfile = pd.ExcelFile(excel)
        print excelfile.sheet_names

# read 11 sheets of
def tocsv(excel, sheet_ids):
    filename = excel[excel.find('FY1'):]
    for i in sheet_ids:
        df = pd.read_excel(excel, sheetname=i)
        # filter out records with empty name
        df = df[pd.notnull(df['Building Number'])]
        outfile = '{0}/csv_FY/sep/{1}_{2}.csv'.format(os.getcwd(), filename[:4], i + 1)
        print 'write to file' + outfile
        df.to_csv(outfile, index=False)

def excel2csv():
    filelist = glob.glob(os.getcwd() + '/input/FY/' + '*.xlsx')
    #filelist = [os.getcwd() + '/input/FY/FY10 data dump.xlsx',
    #            os.getcwd() + '/input/FY/FY12 data dump.xlsx']
    frames = []
    for excel in filelist:
        filename = excel[excel.find('FY1'):]
        print 'processing {0}'.format(filename)
        check_sheetname(excel, False)
        tocsv(excel, range(11))

def df_year(year):
    return [pd.read_csv(os.getcwd() + '/csv_FY/FY{0}_{1}.csv'.format(year, i)) for i in range(1, 12)]

def all_building_set(df_list):
    bd_set_listlist = [[get_building(df) for df in sheet] for sheet in df_list]
    bd_set_list = [reduce(set.union, z) for z in bd_set_listlist]
    return list(reduce(set.union, bd_set_list))

# return a dataframe marking which year of data is available for which building
def mark_bd(df_list, title_list):
    assert(len(df_list) == len(title_list))
    bd_set_listlist = [[get_building(df) for df in x] for x in df_list]
    bd_set_list = [reduce(lambda x, y: x.union(y), z) for z in bd_set_listlist]
    all_bd_set = reduce(lambda x, y: x.union(y), bd_set_list)
    mark_lists = [[1 if x in b else 0 for x in all_bd_set] for b in bd_set_list]
    return pd.DataFrame(dict(zip(title_list, mark_lists)))

def common_building_set(df_list):
    bd_set_listlist = [[get_building(df) for df in sheet] for sheet in df_list]
    bd_set_list = [reduce(set.union, z) for z in bd_set_listlist]
    return list(reduce(set.intersection, bd_set_list))

def region2building():
    print 'create single building energy file ...'
    filelist = glob.glob(os.getcwd() + '/csv_FY/sep/*.csv')
    for csv in filelist:
        df = pd.read_csv(csv)
        year = int(df.ix[0, 'Fiscal Year'])
        bds = set(df['Building Number'].tolist())
        for b in bds:
            df_b = df[df['Building Number'] == b]
            outfile = (os.getcwd() + \
                    '/csv_FY/single/{0}_{1}.csv'.format(b, year))
            print '    {0}_{1}.csv'.format(b, year)
            df_b.to_csv(outfile, index=False)

# region sheet to single building files, in calendar year
def region2building_cal():
    print 'create single building energy file with calendar year ...'
    filelist = glob.glob(os.getcwd() + '/csv_FY/cal/*.csv')
    for csv in filelist:
        df = pd.read_csv(csv)
        year = int(df.ix[0, 'year'])
        bds = set(df['Building Number'].tolist())
        for b in bds:
            df_b = df[df['Building Number'] == b]
            outfile = (os.getcwd() + '/csv_FY/single_cal/{0}_{1}.csv'.format(b, year))
            df_b.to_csv(outfile, index=False)

# fiscal year to calendar year
def fiscal2calyear(y, m):
    if m < 4:
        return y - 1
    else:
        return y

# fiscal month to calendar month
def fiscal2calmonth(m):
    m = m + 9
    if m < 13:
        return m
    else:
        return m % 12

def test_fiscal_convert():
    for year in [2013, 2014, 2015]:
        for month in range(1, 13):
            print '({0}, {1}) --> ({2}, {3})'.format(year,
                                                    month,
                                                    fiscal2calyear(year, month),
                                                    fiscal2calmonth(month))
#test_fiscal_convert()

# note: must process two consecutive FY together
# output FY{year}_{region}.csv files
def fiscal2calendar():
    conn = sqlite3.connect(homedir + 'db/EUAS_input_region_calendar_year.db')
    c = conn.cursor()
    print 'converting fiscal year to calendar year ...'
    for i in range(1, 12):
        # FIXED: original _{0}* will read region 10 info into region 1
        filelist = glob.glob(os.getcwd() + '/csv_FY/sep/*_{0}.csv'.format(i))
        dfs = [pd.read_csv(csv) for csv in filelist]
        df_con = pd.concat(dfs, ignore_index=True)
        df_con['month'] = df_con['Fiscal Month'].map(fiscal2calmonth)
        df_con['year'] = df_con.apply(lambda row: fiscal2calyear(row['Fiscal Year'], row['Fiscal Month']), axis=1)
        # print df_con[['Fiscal Year', 'year', 'Fiscal Month', 'month']].head()
        gr = df_con.groupby('year')
        for name, group in gr:
            yr = str(int(name))[-2:]
            rg = i
            outfile = os.getcwd() + '/csv_FY/cal/CY{0}_{1}.csv'.format(yr, rg)
            print '    writing to CY{0}_{1}.csv'.format(yr, rg)
            group.sort(columns=['Building Number', 'month'],
                       inplace=True)
            group.to_csv(outfile, index=False)
            group.to_sql('CY{0}_{1}'.format(yr, rg), conn,
                         if_exists='replace')
    conn.close()

# deprecated
def building_info():
    filelist = glob.glob(os.getcwd() + '/csv_FY/' + '*.csv')
    dfs13 = df_year(13)
    dfs14 = df_year(14)
    dfs15 = df_year(15)
    df_listlist = [dfs13, dfs14, dfs15]
    print 'number of buildings'
    df = pd.DataFrame({'FY13':check_num_bd(dfs13),
                       'FY14':check_num_bd(dfs14),
                       'FY15':check_num_bd(dfs15)}, index=range(1, 12))
    print df
    df.to_csv(os.getcwd() + '/csv_FY/info/num_building.csv')

    print 'common buildings'
    df2 = pd.DataFrame({'FY13-14': check_common_bd_pair(dfs13, dfs14),
                        'FY14-15': check_common_bd_pair(dfs14, dfs15),
                        'FY13-15': check_common_bd_pair(dfs13, dfs15)},
                       index=range(1, 12))
    df2.to_csv(os.getcwd() + '/csv_FY/info/num_common_building.csv')
    print df2

    common = common_building_set(df_listlist)
    all_bd = all_building_set(df_listlist)
    print 'number of common buildings: {0}'.format(len(common))
    print 'number of all buildings: {0}'.format(len(all_bd))
    df3 = mark_bd(df_listlist, ['2013', '2014', '2015'])
    df3['Building Number'] = all_bd
    df3.to_csv(os.getcwd() + '/csv_FY/info/record_year.csv', index=False)
    print df3

def calculate(calOrFiscal):
    print 'start EUI calculation for {0}scal year ...'.format(calOrFiscal)
    cols = ['Region No.', 'Fiscal Month', 'Fiscal Year',
            'Building Number', 'eui_elec', 'eui_gas', 'eui_oil',
            'eui_steam', 'eui_water', 'eui']
    cols_cat = cols + ['Cat']
    if calOrFiscal == 'cal':
        identifier = 'single_cal'
        out_folder = 'single_eui_cal'
        year_col = 'year'
        month_col = 'month'
        cols += [year_col, month_col]
        cols_cat += [year_col, month_col]
    else:
        identifier = 'single'
        out_folder = 'single_eui'
        year_col = 'Fiscal Year'
        month_col = 'Fiscal Month'
    filelist = glob.glob(os.getcwd() + '/csv_FY/{0}/*.csv'.format(identifier))
    # filelist = [f for f in filelist if 'ME0034ZZ' in f]
    counter = 0
    for csv in filelist:
        df = pd.read_csv(csv)
        filename = csv[csv.find(identifier) + len(identifier) + 1:]
        print counter, filename
        df = df[pd.notnull(df['Gross Sq.Ft'])]
        df_temp = df[df['Gross Sq.Ft'] > 0]
        if len(df_temp) == 0:
            print 'zero floor area in {0}'.format(filename)
            continue
        area = df_temp['Gross Sq.Ft'].tolist()[0]
        df['elec'] = df['Electricity (KWH)'] * 3.412
        df['gas'] = df['Gas (Cubic Ft)'] * 1.026
        df['eui_elec'] = df['elec']/area
        df['eui_gas'] = df['gas']/area
        m_oil = (139 + 138 + 146 + 150)/4
        df['eui_oil'] = df['Oil (Gallon)']/area * m_oil
        df['eui_steam'] = df['Steam (Thou. lbs)']/area * 1194
        df['eui_water'] = df['Water (Gallon)']/area
        df['eui'] = (df['elec'] + df['gas'])/area
        bd = df['Building Number'].tolist()[0]
        yr = int(df[year_col].tolist()[0])
        # note: cols is for pandas v0.13.0, for v.017.0, use columns
        if 'Cat' in df:
            df.to_csv(os.getcwd() + \
                      '/csv_FY/{2}/{0}_{1}.csv'.format(bd, yr,
                                                       out_folder), 
                      cols = cols_cat, index=False)
        else:
            df.to_csv(os.getcwd() + \
                      '/csv_FY/{2}/{0}_{1}.csv'.format(bd, yr,
                                                       out_folder), 
                      cols = cols, index=False)
        counter += 1

def aggregate(year, calOrFiscal):
    print 'aggregating {0} year {1}'.format(calOrFiscal, year)
    if calOrFiscal == 'cal':
        conn = sqlite3.connect(homedir + 'db/eui_calendar.db')
        in_folder = 'single_eui_cal'
        out_folder = 'agg_cal'
        year_col = 'year'
        month_col = 'month'
    else:
        conn = sqlite3.connect(homedir + 'db/eui_fiscal.db')
        in_folder = 'single_eui'
        out_folder = 'agg'
        year_col = 'Fiscal Year'
        month_col = 'Fiscal Month'
    c = conn.cursor()
    filelist = glob.glob(os.getcwd() +
                         '/csv_FY/{1}/*{0}.csv'.format(year, in_folder))
    dfs = []
    for csv in filelist:
        df = pd.read_csv(csv)
        filename = csv[(csv.find(in_folder) + len(in_folder) + 1):]
        # check monthly records availability
        if (len(df) != 12 or len(df[month_col].unique()) != 12):
            print filename
        # change type to string so that no aggregation occur for them
        df['Region No.'] = df['Region No.'].map(lambda x: str(int(x)))
        df['Fiscal Year'] = df['Fiscal Year'].map(lambda x: str(int(x)))
        df['Fiscal Month'] = df['Fiscal Month'].map(lambda x: str(int(x)))
        if calOrFiscal == 'cal':
            df['year'] = df['year'].map(lambda x: str(int(x)))
            df['month'] = df['month'].map(lambda x: str(int(x)))
        region = df.ix[0, 'Region No.']
        yr = df.ix[0, year_col]
        bd = df.ix[0, 'Building Number']
        if calOrFiscal == 'cal':
            if yr != '2012':
                cat = df.ix[len(df) - 1, 'Cat']
            else:
                cat = df.ix[0, 'Cat']
        else:
            if yr != '2013':
                cat = df.ix[0, 'Cat']
            else:
                cat = ''

        df_agg = df.groupby(year_col).sum()
        df_agg['Region No.'] = region
        df_agg['Region No.'] = df_agg['Region No.'].map(lambda x: int(x))
        df_agg[year_col] = yr
        df_agg['Building Number'] = bd
        df_agg['Cat'] = cat
        dfs.append(df_agg)
    df_yr = pd.concat(dfs)
    # print list(df_yr)
    df_yr = df_yr.sort(columns='Region No.')
    df_yr.to_csv(os.getcwd() + \
            '/csv_FY/{1}/eui_{0}.csv'.format(year, out_folder), index=False)
    df_yr.to_sql('eui_{0}'.format(year, out_folder), conn,
                 if_exists='replace')
    conn.close()

# fix me
def aggregate_allyear(calOrFiscal, yearlist):
    if calOrFiscal == 'cal':
        for year in yearlist:
            print year
            aggregate(year, calOrFiscal)
    else:
        for year in yearlist:
            print year
            aggregate(year, calOrFiscal)

def euas2csv():
    df = pd.read_excel(os.getcwd() + '/input/FY/GSA_F15_EUAS_v2.2.xls',
                       sheetname=0)
    program_hd = ['GP', 'LEED', 'first fuel', 'Shave Energy',
                  'GSALink Option(26)', 'GSAlink I(55)', 'E4', 'ESPC',
                  'Energy Star']
    '''
    for hd in program_hd:
        print df[hd].value_counts()
    '''
    df.to_csv(os.getcwd() + '/csv_FY/program/GSA_F15_EUAS.csv', index=False,
              cols=['Building ID', 'GP', 'LEED', 'first fuel', 'Shave Energy',
                    'GSALink Option(26)', 'GSAlink I(55)', 'E4', 'ESPC',
                    'Energy Star'])
    df_bool2int = pd.read_csv(os.getcwd() + '/csv_FY/program/GSA_F15_EUAS.csv')
    for col in program_hd:
        df_bool2int[col] = df_bool2int[col].map(lambda x: 1 if x == '1_Yes'
                                              else 0)
    df_bool2int['Total Programs_v2'] = df_bool2int[program_hd].sum(axis=1)
    df_bool2int['Total Programs (Y/N)_v2'] = df_bool2int['Total Programs_v2'].map(lambda x: 1 if x > 0 else 0)
    df_bool2int.to_csv(os.getcwd() + '/csv_FY/program/GSA_F15_EUAS_int.csv',
                       index=False)

# join EUAS program info and eui info for year 2015
def join_program():
    df_eui = pd.read_csv(os.getcwd() + '/csv_FY/agg_cal/eui_2015.csv')
    df_pro = pd.read_csv(os.getcwd() + '/csv_FY/program/GSA_F15_EUAS_int.csv')
    bd_eui = set(df_eui['Building Number'].tolist())
    bd_pro = set(df_pro['Building ID'].tolist())
    print 'number of buildings in eui_2015: {0}'.format(len(bd_eui))
    print 'number of buildings in program : {0}'.format(len(bd_pro))
    print 'number of common buildings: {0}'.format(len(bd_eui.intersection(bd_pro)))
    print 'buildings left out:{0}'.format(bd_eui.difference(bd_pro))
    df_merge = pd.merge(df_eui, df_pro, how='left', left_on='Building Number',
                        right_on = 'Building ID')
    df_merge.info()
    df_merge.drop('Building ID', inplace=True, axis=1)
    df_merge.fillna(0, inplace=True)
    df_merge.to_csv(os.getcwd() + '/csv_FY/join_cal/join_2015.csv', index=False)

def report_false():
    filelist = glob.glob(os.getcwd() + '/csv_FY/agg/*.csv')
    for csv in filelist:
        df = pd.read_csv(csv)
        yr = int(df.ix[0, 'Fiscal Year'])
        df_eui = df[df['eui'] < 20]
        df_water = df[df['eui_water'] < 5]
        outfile_eui = os.getcwd() + '/csv_FY/false_eui/false_eui_{0}.csv'.format(yr)
        print (yr, 'false eui', len(set(df_eui['Building Number'].tolist())))
        print (yr, 'false water', len(set(df_water['Building Number'].tolist())))
        outfile_water = os.getcwd() + '/csv_FY/false_eui/false_water_{0}.csv'.format(yr)
        df_eui.to_csv(outfile_eui, index=False)
        df_water.to_csv(outfile_water, index=False)

def report_false_15():
    df = pd.read_csv(os.getcwd() + '/csv_FY/join/join_2015.csv')
    df_eui = df[df['eui'] < 20]
    outfile = os.getcwd() + '/csv_FY/false_eui/false_eui_2015.csv'
    df_eui.to_csv(outfile, index=False)
    print 'false eui:'
    false_bd_eui = df_eui['Building Number'].tolist()
    for item in false_bd_eui:
        print item

    df_water = df[df['eui_water'] < 5]
    outfile = os.getcwd() + '/csv_FY/false_eui/false_water_2015.csv'
    df_water.to_csv(outfile, index=False)
    print 'false water:'
    false_wt_eui = df_eui['Building Number'].tolist()
    for item in false_wt_eui:
        print item

# bookmark
def weather_dict(criteria):
    '''
    df = pd.read_excel(os.getcwd() + '/input/FY/EUAS Data_Oct 2014 To Aug 2015.xlsx', sheetname=0)
    df.to_csv(os.getcwd() + '/csv_FY/weather.csv')
    '''
    df_weather = pd.read_csv(os.getcwd() + '/csv_FY/weather.csv')
    df_weather = df_weather[['Building Number', 'Weather Station']]
    weather_station = set(df_weather['Building Number'].tolist())
    #print list(set(df_weather['Weather Station'].tolist()))

    '''
    for csv in criteria == 'eui':
        files = glob.glob(os.getcwd() + '/csv_FY/false_eui/false_eui_{0}.csv'.format(year))
        false_bd_set_list = [set((pd.read_csv(csv))['Building Number'].tolist()) for csv in files]
        false_bd_set = reduce(set.union, false_bd_set_list)
    elif criteria == 'all':
        files = glob.glob(os.getcwd() + '/csv_FY/false_eui/*_{0}.csv'.format(year))
        false_bd_set_list = [set((pd.read_csv(csv))['Building Number'].tolist()) for csv in files]
        false_bd_set = reduce(set.union, false_bd_set_list)
    else:
        false_bd_set = set([])

    filelist = glob.glob(os.getcwd() + '/csv_FY/agg/*.csv')

    for csv in filelist:
        df = pd.read_csv(csv)
        df['bad'] = df['Building Number'].map(lambda x: 1 if x in false_bd_set else 0)
        df = df[df['bad'] == 0]
        bds = get_building(df)
        yr = df.ix[0, 'Fiscal Year']
        print '{0}, num_building: {1}, common_building: {2}'.format(yr, len(bds), len(bds.intersection(weather_station)))
    '''
def get_fuel_type(years):
    print 'get fuel type ...'
    year_labels = [str(y)[2:] for y in years]
    for y in year_labels:
        df = pd.read_csv(os.getcwd() +
                         '/csv_FY/raw_concat/FY{0}.csv'.format(y))
        df_sum = df.groupby('Building Number').sum()
        cols = ['Electricity (KWH)', 'Steam (Thou. lbs)', 
                'Gas (Cubic Ft)', 'Oil (Gallon)', 
                'Chilled Water (Ton Hr)']
        rename_cols = {'Electricity (KWH)': 'Electricity', 
                       'Steam (Thou. lbs)': 'Steam',
                       'Gas (Cubic Ft)': 'Gas', 
                       'Oil (Gallon)': 'Oil',
                       'Chilled Water (Ton Hr)': 'Chilled Water'}
        df_sum = df_sum[cols]
        for col in cols:
            df_sum[col] = df_sum[col].map(lambda x: 1 if x > 0 else 0)
        df_sum['num_heat_fuel'] = df_sum['Steam (Thou. lbs)'] + \
                                  df_sum['Gas (Cubic Ft)'] + \
                                  df_sum['Oil (Gallon)']
        df_sum.rename(columns = rename_cols, inplace=True)
        df_sum['None (all electric?)'] = df_sum['num_heat_fuel'].map(lambda x: 1 if x == 0 else 0)
        df_sum['Gas Only'] = df_sum.apply(lambda r: 1 if r['Gas'] == 1 and r['num_heat_fuel'] == 1 else 0, axis=1)
        df_sum['Oil Only'] = df_sum.apply(lambda r: 1 if r['Oil'] == 1 and r['num_heat_fuel'] == 1 else 0, axis=1)
        df_sum['Steam Only'] = df_sum.apply(lambda r: 1 if r['Steam'] == 1 and r['num_heat_fuel'] == 1 else 0, axis=1)
        df_sum['Gas + Oil'] = df_sum['Gas'] & df_sum['Oil']
        df_sum['Gas + Steam'] = df_sum['Gas'] & df_sum['Steam']
        df_sum['Oil + Steam'] = df_sum['Oil'] & df_sum['Steam']
        df_sum['Gas + Oil + Steam'] = df_sum['Oil'] & \
                                      df_sum['Steam'] & \
                                      df_sum['Gas']
        for col in ['Gas + Oil', 'Gas + Steam', 'Oil + Steam', 
                    'Gas + Oil + Steam', 'Gas Only', 'Oil Only', 
                    'Steam Only']:
            df_sum[col] = df_sum[col].apply(lambda x: 1 if x else 0)
        df_sum.to_csv(os.getcwd() +
                      '/csv_FY/fuel_type/FY{0}.csv'.format(y))

def fuel_type_plot(years, catfilter):
    print 'plot fuel type {0} to {1}, {2}'.format(years[0], years[-1], catfilter)
    year_labels = [str(y)[2:] for y in years]
    filelist = ['{0}fuel_type/FY{1}.csv'.format(homedir, yr) for yr in year_labels]
    dfs = []
    for f in filelist:
        df = pd.read_csv(f)
        year = f[-6: -4]
        df['year'] = '20{0}'.format(year)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=False)
    def select_col(r, cols):
        values = [r[c] for c in cols]
        return cols[values.index(1)]
    df_all.rename(columns={'None (all electric?)': 'No Data'},
                  inplace=True)
    fuel_type_cols = ['No Data', 'Gas Only', 'Oil Only', 
                      'Steam Only', 'Gas + Oil', 'Gas + Steam', 
                      'Oil + Steam', 'Gas + Oil + Steam']
    df_all['Heating Fuel Type'] = \
        df_all.apply(lambda r: select_col(r, fuel_type_cols), axis=1)
    df_all = df_all[['Building Number', 'Heating Fuel Type', 'year']]
    df_cat = pd.read_csv(homedir + 'master_table/static_info.csv')
    df_cat = df_cat[['Building Number', 'Cat_latest']]
    df_all2 = pd.merge(df_all, df_cat, on='Building Number',
                       how='left')
    if catfilter == 'AI':
        df_all2 = df_all2[df_all2['Cat_latest'].isin(['A', 'I'])]
    sns.set_style("white")
    sns.set_palette(sns.color_palette('Set2'))
    sns.set_context("talk", font_scale=1.0)
    sns.mpl.rc("figure", figsize=(10, 5))
    sns.countplot(x='year', order= [str(x) for x in years],
                  hue='Heating Fuel Type', palette='Set3',
                  hue_order=fuel_type_cols, data=df_all2)
    plt.legend(loc = 2, bbox_to_anchor=(1, 1))
    my_dpi=300
    plt.title('Heating Fuel Type Count (FY{0} - FY{1})'.format(years[0], years[-1]))
    plt.suptitle('Building category: {0}'.format(catfilter))
    plt.ylabel('Number of Buildings')
    plt.xlabel('Fiscal Year')
    P.savefig(os.getcwd() + '/plot_FY_annual/fuel_type_{0}.png'.format(catfilter), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi), bbox_inches='tight')
    plt.close()
    
# join fuel types to filter bit
def join_fueltype(years):
    print 'join fuel type ...'
    year_labels = [str(y)[2:] for y in years]
    indicator_df = pd.read_csv(homedir + 'filter_bit/fis/indicator_all.csv')

    cols = ['None (all electric?)', 'Gas Only', 'Oil Only', 
            'Steam Only', 'Gas + Oil', 'Gas + Steam', 
            'Oil + Steam', 'Gas + Oil + Steam', 'Chilled Water']
    dfs = []
    for yr in year_labels:
        df = pd.read_csv('{0}fuel_type/FY{1}.csv'.format(homedir, yr))
        df = df[['Building Number'] + cols]
        newcols = ['{0}_{1}'.format(x, yr) for x in cols]
        df.rename(columns=dict(zip(cols, newcols)), inplace=True)
        dfs.append(df)
    df_all = reduce(lambda x, y: pd.merge(x, y, on='Building Number',
                                          how='left'),
                    [indicator_df] + dfs)
    for c in cols:
        df_all['sum_{0}'.format(c)] = df_all[[x for x in list(df_all) if c in x]].sum(axis=1)
        df_all[c] = df_all['sum_{0}'.format(c)].map(lambda x: 1 if x == 5 else 0)

    sum_cols = ['sum_{0}'.format(c) for c in cols]
    df_all.drop(sum_cols, axis=1, inplace=True)
    df_all.to_csv(homedir + 'filter_bit/fis/indicator_all_fuel.csv', index=False)
    df_all = df_all[df_all['Cat_latest'].isin(['A', 'I'])]
    df_all.to_csv(homedir + 'filter_bit/fis/indicator_all_fuel_catAI.csv', index=False)
    return

def energy_info(yearlist):
    conn = sqlite3.connect(homedir + 'db/energy_info.db')
    c = conn.cursor()
    print '    generating energy_info.csv master table ...'
    filelist = [homedir + 'raw_concat/FY{0}.csv'.format(str(x)[2:])
                for x in yearlist]
    dfs = []
    # take the average of No.1 to 6
    m_oil = (139 + 138 + 146 + 150)/4
    for f in filelist:
        filename = f[f.rfind('/') + 1:]
        df = pd.read_csv(f)
        df = df[['Building Number', 'Fiscal Year', 'Fiscal Month',
                 'Gross Sq.Ft', 'Electricity (KWH)', 
                 'Steam (Thou. lbs)', 'Gas (Cubic Ft)', 
                 'Oil (Gallon)', 'Chilled Water (Ton Hr)', 
                 'Water (Gallon)']]
        df['Electricity (kBtu)'] = df['Electricity (KWH)'] * 3.412
        df['Gas (kBtu)'] = df['Gas (Cubic Ft)'] * 1.026
        df['Oil (kBtu)'] = df['Oil (Gallon)'] * m_oil
        df['Steam (kBtu)'] = df['Steam (Thou. lbs)'] * 1194
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all['month'] = df_all['Fiscal Month'].map(fiscal2calmonth)
    df_all['year'] = df_all.apply(lambda row: fiscal2calyear(row['Fiscal Year'], row['Fiscal Month']), axis=1)
    df_all.sort(columns=['Building Number', 'Fiscal Year', 
                         'Fiscal Month'], inplace=True)
    df_all = df_all[['Building Number', 'Fiscal Year', 'Fiscal Month',
                     'year', 'month', 'Gross Sq.Ft', 'Electricity' + 
                     ' (kBtu)', 'Gas (kBtu)', 'Oil (kBtu)', 'Steam' + 
                     ' (kBtu)', 'Water (Gallon)', 'Chilled Water (Ton'
                     ' Hr)']]
    df_all.to_csv(homedir + 'master_table/energy_info_monthly.csv', index=False)
    df_all.to_sql('energy_info_monthly', conn, if_exists='replace')
    df_zero = df_all[df_all['Gross Sq.Ft'] == 0]
    df_zero = df_zero[['Building Number', 'Fiscal Year', 'Fiscal '
        'Month', 'Gross Sq.Ft']]
    df_zero.to_csv(homedir + 'master_table/zero_area_entries.csv',
                   index=False)
    df_all = df_all[df_all['Gross Sq.Ft'] > 0]
    df_all['eui_elec'] = df_all['Electricity (kBtu)']/df_all['Gross '
        'Sq.Ft']
    df_all['eui_gas'] = df_all['Gas (kBtu)']/df_all['Gross Sq.Ft']
    df_all['eui_oil'] = df_all['Oil (kBtu)']/df_all['Gross Sq.Ft']
    df_all['eui_steam'] = df_all['Steam (kBtu)']/df_all['Gross Sq.Ft']
    df_all['eui_water'] = df_all['Water (Gallon)']/df_all['Gross '
        'Sq.Ft']
    df_all['eui'] = (df_all['eui_elec'] + df_all['eui_gas'])
    df_all.to_csv(homedir + 'master_table/energy_eui_monthly.csv', index=False)
    df_all.to_sql('energy_eui_monthly', conn, if_exists='replace')
    df_eui_fy = df_all.groupby(['Building Number', 'Fiscal Year'],
                               as_index=False).sum()
    df_eui_fy = df_eui_fy[['Building Number', 'Fiscal Year',
                           'eui_elec', 'eui_gas', 'eui_oil',
                           'eui_steam', 'eui_water', 'eui']]
    df_eui_fy.to_csv(homedir + 'master_table/eui_by_fy.csv',
                     index=False)
    df_eui_fy.to_sql('eui_by_fy', conn, if_exists='replace')
    df_eui_cy = df_all.groupby(['Building Number', 'year'],
                               as_index=False).sum()
    df_eui_cy = df_eui_cy[['Building Number', 'year',
                           'eui_elec', 'eui_gas', 'eui_oil',
                           'eui_steam', 'eui_water', 'eui']]
    df_eui_cy.to_csv(homedir + 'master_table/eui_by_cy.csv',
                     index=False)
    df_eui_cy.to_sql('eui_by_cy', conn, if_exists='replace')
    conn.close()
    return

def join_static():
    conn = sqlite3.connect(homedir + 'db/static_info_tidy.db')
    c = conn.cursor()
    print '    creating static_info.csv master table ...'
    df1 = pd.read_csv(homedir + 'master_table/EUAS_static_tidy.csv')
    df2 = pd.read_csv(os.getcwd() + '/input/FY/static info/Entire GSA Building Portfolio.csv')
    df2.to_sql('Entire_GSA_Building_Portfolio_input', conn, if_exists='replace')
    df2 = df2[['Building ID', 'Street', 'City', 'Zip Code']]
    df2.rename(columns={'Building ID': 'Building Number', 'Street':
                        'Street Address'}, inplace=True)
    df2['Zip Code'] = df2['Zip Code'].map(lambda x: x[:5])
    df2['source'] = 'Entire_GSA_Building_Portfolio_input'
    filename = os.getcwd() + '/csv/all_column/sheet-0-all_col.csv'
    df_use = pd.read_csv(filename)
    df_use.to_sql('PortfolioManager_sheet0_input', conn,
                  if_exists='replace')
    df_use = df_use[['Property Name', 'Street Address',
                     'City/Municipality', 'Postal Code']]
    df_use['Property Name'] = df_use['Property Name'].map(lambda x: x.partition(' ')[0][:8])
    df_use['Postal Code'] = df_use['Postal Code'].map(lambda x: x[:5])
    df_use.rename(columns={'Property Name': 'Building Number',
                           'City/Municipality': 'City', 'Postal Code':
                           'Zip Code'}, inplace=True)
    df_use.drop_duplicates(cols='Building Number', inplace=True)
    df_use['source'] = 'PortfolioManager_sheet0_input'
    df_loc = pd.concat([df2, df_use], ignore_index=True)
    df_loc.sort(columns=['Building Number'], inplace=True)
    df_loc.to_csv(homedir + 'master_table/building_address_source.csv',
                  index=False)
    df_loc.to_sql('building_address_source', conn, if_exists='replace')
    df_loc.drop_duplicates(cols=['Building Number', 'Street Address',
                                 'City', 'Zip Code'], inplace=True)
    # print (df_loc.groupby('Building Number').filter(lambda x: len(x) > 1))
    df_loc.to_csv(homedir + \
                  'master_table/building_address_source_dropdup.csv',
                  index=False)
    # df_loc.drop_duplicates(cols=['Building Number'], inplace=True)
    df_all = pd.merge(df1, df_loc, how='left', on='Building Number')
    df_all.to_csv(homedir + 'master_table/EUAS_address.csv',
                  index=False)
    df_all.to_sql('EUAS_address', conn, if_exists='replace')
    df_all.drop_duplicates(cols=['Building Number'], inplace=True)
    df_all.to_csv(homedir + 'master_table/EUAS_address_dropdupe.csv',
                  index=False)
    df_all.to_sql('EUAS_address_dropdupe', conn, if_exists='replace')
    df_no_address = df_all[df_all['Street Address'].isnull()]
    df_no_address.to_csv(homedir + 'master_table/EUAS_no_address.csv',
                         index=False)
    df_no_address.to_sql('EUAS_no_address', conn, if_exists='replace')
    conn.close()
    return

def join_static_ecm():
    print '    join static_info.csv with ecm_highlevelAction.csv ...'
    df1 = pd.read_csv(homedir + 'master_table/static_info.csv')
    df2 = pd.read_csv(homedir + \
                      'master_table/ECM/ecm_highlevelAction.csv')
    df2.rename(columns={'Building ID': 'Building Number'},
               inplace=True)
    df = pd.merge(df1, df2, on='Building Number', how='left')
    df.to_csv(homedir + 'master_table/static_info_wECM.csv',
              index=False)
    conn = sqlite3.connect(homedir + 'db/static_info.db')
    c = conn.cursor()
    df.to_sql('static_info_wECM', conn, if_exists='replace')
    conn.close()
    return

def join_indicator_static_all(filtercols):
    df1 = pd.read_csv(homedir + 'master_table/static_info_wECM.csv')
    df2 = pd.read_csv(homedir + 'filter_bit/fis/indicator_all.csv')
    # ATTENTION!!!!!
    df2 = df2[['Building Number'] + filtercols]
    df = pd.merge(df1, df2, on='Building Number', how='left')
    print 'write to "static_info_wECM_indicator" ...'
    df.to_csv(homedir + 'master_table/static_info_wECM_indicator.csv',
              index=False)
    df3 = pd.read_csv(homedir + 'join/join_2015_proonly.csv')
    df3 = df3[['Building Number', 'GP', 'LEED', 'first fuel', 
               'Shave Energy', 'E4', 'ESPC', 'GP_only', 'LEED_only', 
               'first fuel_only', 'Shave Energy_only', 'E4_only',
               'ESPC_only', 'GSALink_only']]
    df = df[['Building Number', 'Cat_latest', 'GSALink', 
             'Advanced Metering', 'Building Envelope' , 
             'Building Tuneup or Utility Improvements' ,
             'HVAC' , 'IEQ' , 'Lighting'] + filtercols]
    df_all = pd.merge(df, df3, how='left', on='Building Number')
    print 'write to "static_info_ECM_program" ...'
    df_all.to_csv(homedir + \
                  'master_table/static_info_ECM_program.csv',
                  index=False)
    return
    
def plot_action_by_prog():
    join_indicator_static_all(['has_data >= 8 years', 
                               'good_both >= 8 years', 'good_area_15'])
    df = pd.read_csv(homedir + \
                     'master_table/static_info_ECM_program.csv')
    df = df[df['good_area_15'] == 1]
    df.rename(columns={'GP_only': 'GSA Guiding Principles_only', 
                       'first fuel_only':
                       'first fuel'.title() + '_only'}, inplace=True)
    # programs = ['GSA Guiding Principles', 'LEED', 'First Fuel', 
    #             'Shave Energy', 'E4', 'ESPC', 'GSALink']

    programs = ['GSA Guiding Principles_only', 'LEED_only', 
                'First Fuel_only', 'Shave Energy_only', 'E4_only', 
                'ESPC_only', 'GSALink_only']
    actions = ['Advanced Metering', 'Building Envelope', 
               'Building Tuneup or Utility Improvements', 'HVAC', 
               'IEQ', 'Lighting']
    dfs = []
    for p in programs:
        df1 = df[df[p] == 1]
        df1['program'] = p
        for a in actions:
            df2 = df1[df1[a] == 1]
            df2['action'] = a
            dfs.append(df2)
    df_all = pd.concat(dfs, ignore_index=False)
    my_dpi = 300
    sns.set_palette(sns.color_palette('Set3'))
    sns.set_context("talk", font_scale=1.2)
    sns.countplot(y='program', hue='action', data=df_all)
    plt.ylabel('Program')
    plt.xlabel('Building Count')
    plt.legend(loc = 2, bbox_to_anchor=(1, 1))
    plt.title('Building ECM Action by Energy Program (Solo) Plot')
    plt.suptitle('Plot set: All building with energy data in FY2015')
    P.savefig(os.getcwd() + '/plot_FY_annual/action_by_program_only.png', dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi), bbox_inches='tight')
    plt.close()

def get_time_filter_label(timerange):
    tokens = timerange.split(' ')
    if len(tokens) < 1:
        print 'illegal time range expression'
        return None
    elif len(tokens) == 1:
        ab = int(tokens[0][2:])
        return (ab, ab)
    elif len(tokens) == 2:
        if token[0] == 'before':
            b = int(tokens[1][2:])
            return (None, b)
        elif token[0] == 'after':
            a = int(tokens[1][2:])
            return (a, None)
        else:
            print 'illegal time range expression'
            return None
    elif len(tokens) == 5:
        b = int(tokens[1][2:])
        a = int(tokens[4][2:])
        return (a, b)
    else:
        print 'illegal time range expression'
        return None

# FIXME, use timerange to calculate filter
def plot_static(plotset, range_cat, range_function, range_ecm, title_cat, title_function, title_ecm):
    df = pd.read_csv(homedir + 'master_table/static_info_wECM_indicator.csv')
    office_list = get_office()
    if plotset == 'office':
        df2 = df2[df2['Building Number'].isin(office_list)]

    my_dpi = 300
    sns.set_palette(sns.color_palette('Set2'))
    sns.set_context("talk", font_scale=1.2)
    
    df2 = df[df[range_function] == 1]
    df2.info()
    df2.fillna({'Self-Selected Primary Function': 'No Data'}, inplace=True)
    df2['with ECM Action'] = df2['number of ECM action'].map(lambda x: 1 if x > 0 else 0)
    # print df2['Self-Selected Primary Function'].value_counts()
    sns.countplot(y='Self-Selected Primary Function', data=df2,
                  orient='v')
    plt.title('Building Type Count Plot')
    plt.suptitle('plot set: {0} building {1}'.format(plotset, title_function))
    plt.ylabel('Self-Selected Primary Function')
    plt.xlabel('Building Count')
    P.savefig(os.getcwd() + '/plot_FY_annual/use_count_from_pm.png', dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi), bbox_inches='tight')
    plt.close()
    df3 = df2[['Self-Selected Primary Function', 'with ECM Action']]
    print df3.groupby(['Self-Selected Primary Function', 'with ECM Action']).count()

    df2 = df[df[range_cat] == 1]
    df.rename(columns={'Cat_latest': 'Category'}, inplace=True)
    df['Category'].replace('False', 'No Data', inplace=True)
    # print df['Cat'].value_counts()
    # # temp
    # df = df[df['Cat'].isin(['A', 'I'])]
    # df = df[df['with ECM action'] == 1]
    rename_dict = {\
        'A': 'A\n{0}'.format('\n'.join(tw.wrap('Government Goal', 10))), 
        'I': 'I\n{0}'.format('\n'.join(tw.wrap('Energy Intensive', 10))), 
        'B': 'B\n{0}'.format('\n'.join(tw.wrap('Government Exempt', 10))), 
        'C': 'C\n{0}'.format('\n'.join(tw.wrap('Lease', 10))), 
        'D': 'D\n{0}'.format('\n'.join(tw.wrap('Lease Exempt', 10))), 
                   'E': 'E\n{0}'.format('\n'.join(tw.wrap('Reimbursable non reportable', 12)))}

    df.replace(rename_dict, inplace=True)
    order = [rename_dict[k] for k in ['A', 'I', 'B', 'C', 'D', 'E']]
    sns.countplot(x='Category', order=order, data=df)
    plt.title('Building Category Count Plot')
    plt.suptitle('plot set: {0} building {1}'.format(plotset, range_cat))
    plt.ylabel('Building Count')
    P.savefig(os.getcwd() + '/plot_FY_annual/cat_count_from_euas.png', dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi), bbox_inches='tight')
    plt.close()

    df2 = df[df[range_ecm] == 1]
    df_ecm = df.copy()
    df_ecm['number of ECM action'] = df_ecm['number of ECM action'].map(lambda x: int(x) if not np.isnan(x) else 0)
    df_ecm = df[df['number of ECM action'] > 0]
    sns.countplot(x='number of ECM action', order=range(1, 7), data=df_ecm)
    plt.title('Building ECM Action Count Plot')
    plt.ylabel('Building Count')
    plt.suptitle('plot set: {0} building {1}'.format(plotset, title_ecm))
    P.savefig(os.getcwd() + '/plot_FY_annual/{0}_ecm_count.png'.format(plotset), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi), bbox_inches='tight')
    plt.close()

    ECMs = ['Advanced Metering', 'Building Envelope', 
            'Building Tuneup or Utility Improvements', 
            'HVAC', 'IEQ', 'Lighting']
    dfs = []
    for col in ECMs:
        df_temp = df.copy()
        df_temp = df_temp[df_temp[col] == 1]
        df_temp['ECM Actions'] = '\n'.join(tw.wrap(col, 12))
        dfs.append(df_temp)
    df_all = pd.concat(dfs, ignore_index=True)
    print df_all['ECM Actions'].head()
    # df_all.rename(columns={'count': 'Building Count'}, inplace=True)
    sns.countplot(x='ECM Actions', data=df_all)
    plt.title('ECM high level action count plot')
    plt.ylabel('Building Count')
    plt.suptitle('plot set: {0} building {1}'.format(plotset, title_ecm))
    P.savefig(os.getcwd() + '/plot_FY_annual/{0}_ecmCount.png'.format(plotset), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi), bbox_inches='tight')
    plt.close()
    return

ylabel_dict = {'eui':'Electricity + Gas [kBtu/sq.ft]',
               'eui_elec':'Electric EUI [kBtu/sq.ft]',
               'eui_gas':'Gas EUI [kBtu/sq.ft]',
               'eui_oil':'Oil EUI [Gallons/sq.ft]',
               'eui_water':'WEUI [Gallons/sq.ft]'}

filter_dict = {'eui':'EUI Electric >= 12 and EUI Gas >= 3',
               'eui_elec':'Electric >= 12',
               'eui_gas':'EUI Gas >= 3',
               'eui_water':'WEUI >= 5'}

title_dict = {'eui':'EUI', 'eui_elec':'Electric EUI',
              'eui_gas':'Gas EUI', 'eui_oil':'Oil EUI',
              'eui_water':'WEUI'}

def program_info():
    df = pd.read_csv(homedir + 'join/join_2015_proonly.csv')
    df.rename(columns={'GP': 'GSA Guiding Principles', 'first fuel': 'first fuel'.title()}, inplace=True)
    df.replace(0, np.nan, inplace=True)
    df.info()

def plot_program(plotset):
    df = pd.read_csv(homedir + 'join/join_2015_proonly.csv')
    df.rename(columns={'Total Programs_v2': 'Number of ECM Program', 
                       'Total Programs (Y/N)_v2': 'Has ECM Program'},
              inplace=True)
    office_list = get_office()
    if plotset == 'office':
        df = df[df['Building Number'].isin(office_list)]
    counts = (df['Has ECM Program'].value_counts())
    df['Has ECM Program'].replace(1, 'Yes (n = {0})'.format(counts[1]), inplace=True)
    df['Has ECM Program'].replace(0, 'No (n = {0})'.format(counts[0]), inplace=True)
    df['Number of ECM Program'] = df['Number of ECM Program'].map(lambda x: int(x))
    my_dpi = 300
    sns.set_palette(sns.color_palette('Set2'))
    sns.set_context("talk", font_scale=1.2)
    sns.countplot(y='Number of ECM Program', order=range(7), data=df)
    plt.title('ECM Program Count Plot')
    plt.suptitle('Plot set: {0} building with energy data in {1}'.format(plotset, timerange))
    P.savefig(os.getcwd() + '/plot_FY_annual/{0}_pro_count_from_euas15.png'.format(plotset), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi), bbox_inches='tight')
    plt.close()
    # BOOKMARK: plot program
    df.rename(columns={'GP': 'GSA Guiding Principles', 'first fuel': 'first fuel'.title()}, inplace=True)
    programs = ['GSA Guiding Principles', 'LEED', 'First Fuel', 
                'Shave Energy', 'E4', 'ESPC', 'GSALink']
    dfs = []
    for col in programs:
        df_temp = df.copy()
        df_temp = df_temp[df_temp[col] == 1]
        df_temp['ECM Program'] = '\n'.join(tw.wrap(col, 12))
        dfs.append(df_temp)
    df_all = pd.concat(dfs, ignore_index=True)
    # df_all.rename(columns={'count': 'Building Count'}, inplace=True)
    sns.countplot(x='ECM Program', data=df_all)
    plt.title('ECM program count plot')
    plt.ylabel('Building Count')
    plt.suptitle('Plot set: {0} building with energy data in {1}'.format(plotset, timerange))
    P.savefig(os.getcwd() + '/plot_FY_annual/{0}_pro_from_euas15.png'.format(plotset), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi), bbox_inches='tight')
    plt.close()

def plot_program_yesno(theme, plotset):
    df = pd.read_csv(homedir + 'join/join_2015_proonly.csv')
    df.rename(columns={'Total Programs_v2': 'Number of ECM Program', 
                       'Total Programs (Y/N)_v2': 'Has ECM Program'},
              inplace=True)
    df['Number of ECM Program'] = df['Number of ECM Program'].map(lambda x: int(x))
    office_list = get_office()
    if plotset == 'office':
        df = df[df['Building Number'].isin(office_list)]
    if theme == 'eui':
        df = df[df['eui_elec'] >= 12]
        df = df[df['eui_gas'] >= 3]
    if theme == 'eui_gas':
        df = df[df[theme] >= 3]
    if theme == 'eui_elec':
        df = df[df[theme] >= 12]
    if theme == 'eui_water':
        df = df[df['eui_water'] >= 5]
    counts = (df['Has ECM Program'].value_counts())
    medians = (df.groupby('Has ECM Program').median())[theme]
    print medians
    reduction = round((medians[0] - medians[1]) / medians[0] * 100, 0)
    df['Has ECM Program'].replace(1, 'Yes (n = {0})'.format(counts[1]), inplace=True)
    df['Has ECM Program'].replace(0, 'No (n = {0})'.format(counts[0]), inplace=True)
    my_dpi = 300
    sns.set_style("white")
    sns.set_palette(sns.color_palette('Set2'))
    sns.set_context("talk", font_scale=1.2)
    sns.boxplot(x='Has ECM Program', y=theme, data=df)
    sns.stripplot(x='Has ECM Program', y=theme, data=df,
                  jitter=0.3, edgecolor='gray',
                  color = 'gray', size=2, alpha=0.5)
    plt.title('{0}% less {1} consumption, With vs Without ECM Program'.format(reduction, title_dict[theme]))
    plt.ylabel(ylabel_dict[theme])
    plt.suptitle('FY2015 EUAS {2} building data set with positive sq.ft.\n{0} (n = {1})'.format(filter_dict[theme], len(df), plotset))
    if theme == 'eui_water':
        plt.ylim((0, 60))
    else:
        plt.ylim((0, 140))
    P.savefig(os.getcwd() + '/plot_FY_annual/{1}_prog_melt_{0}.png'.format(theme, plotset), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi), bbox_inches='tight')
    plt.close()

def plot_program_energy(plotset):
    import plot_dist_wtno as pdw
    themes = ['eui', 'eui_elec', 'eui_gas', 'eui_water']
    ylims = [140, 140, 140, 40]
    program = ['LEED', 'GP', 'GSALink', 'first fuel', 'E4', 
               'Shave Energy','ESPC', 'Energy Star']
    inputfile = os.getcwd() + '/csv_FY/join/join_2015_proonly.csv'
    prefix = 'proEnergy'
    for theme, ylim in zip(themes, ylims):
         pdw.plot_box_vio(inputfile, program, prefix, theme, ylim,
                          False, 'plus')
         pdw.plot_box_vio(inputfile, program, prefix, theme, ylim,
                          True, 'plus')

# deprecate
def plot_program_energy_dep(theme, plotset):
    df = pd.read_csv(homedir + 'join/join_2015_proonly.csv')
    df.rename(columns={'Total Programs_v2': 'Number of ECM Program', 
                       'Total Programs (Y/N)_v2': 'Has ECM Program'},
              inplace=True)
    df.rename(columns={'GP': 'GSA Guiding Principles', 'first fuel': 'first fuel'.title()}, inplace=True)
    df['Number of ECM Program'] = df['Number of ECM Program'].map(lambda x: int(x))
    office_list = get_office()
    if plotset == 'office':
        df = df[df['Building Number'].isin(office_list)]
    if theme == 'eui':
        df = df[df['eui_elec'] >= 12]
        df = df[df['eui_gas'] >= 3]
    if theme == 'eui_gas':
        df = df[df[theme] >= 3]
    if theme == 'eui_elec':
        df = df[df[theme] >= 12]
    if theme == 'eui_water':
        df = df[df['eui_water'] >= 5]
    programs = ['GSA Guiding Principles', 'LEED', 'First Fuel', 'Shave Energy', 'E4', 'ESPC', 'GSALink']
    dfs = []
    for col in programs:
        df_temp = df.copy()
        df_temp['program'] = '\n'.join(tw.wrap(col, 15))
        df_temp['status'] = df[col].map(lambda x: 'Yes' if x == 1 else
                                        'No')
        dfs.append(df_temp)
    df_all = pd.concat(dfs, ignore_index=True)
    sns.set_style("white")
    sns.set_palette(sns.color_palette('Set2'))
    sns.set_context("talk", font_scale=1.2)
    g = sns.FacetGrid(df_all, col='program', size=5, aspect=0.5)
    g = g.map(sns.boxplot, 'status', theme)
    g.set(ylim=(0, 140))
    plt.show()

    # sns.mpl.rc("figure", figsize=(10, 5))
    # sns.boxplot(x='program', y=theme, hue='status', data=df_all)
    # sns.stripplot(x='program', y=theme, hue='status', data=df_all, jitter=0.3, edgecolor='gray', color = 'gray', size=2, alpha=0.5)
    # plt.ylim((0, 140))
    # plt.legend(loc = 2, bbox_to_anchor=(1, 1))
    plt.show()

def plot_program_num(theme, plotset):
    df = pd.read_csv(homedir + 'join/join_2015_proonly.csv')
    df.rename(columns={'Total Programs_v2': 'Number of ECM Program', 
                       'Total Programs (Y/N)_v2': 'Has ECM Program'},
              inplace=True)
    df['Number of ECM Program'] = df['Number of ECM Program'].map(lambda x: int(x))
    office_list = get_office()
    if plotset == 'office':
        df = df[df['Building Number'].isin(office_list)]
    if theme == 'eui':
        df = df[df['eui_elec'] >= 12]
        df = df[df['eui_gas'] >= 3]
    if theme == 'eui_gas':
        df = df[df[theme] >= 3]
    if theme == 'eui_elec':
        df = df[df[theme] >= 12]
    if theme == 'eui_water':
        df = df[df['eui_water'] >= 5]
    counts = (df['Number of ECM Program'].value_counts())
    count_labels = []
    for i in range(7):
        count_lb = '{0}\n(n = {1})'.format(i, counts[i])
        df['Number of ECM Program'].replace(i, count_lb, inplace=True)
        count_labels.append(count_lb)
    my_dpi = 300
    sns.set_style("white")
    sns.set_palette(sns.color_palette('Set2'))
    sns.set_context("talk", font_scale=1.2)
    sns.boxplot(x='Number of ECM Program', y=theme, order=count_labels,
                data=df)
    sns.stripplot(x='Number of ECM Program', y=theme,
                  order=count_labels, data=df, jitter=0.3,
                  edgecolor='gray', color = 'gray', size=2, alpha=0.5)
    plt.title('{0} consumption plot by the number of ECM program'.format(title_dict[theme]))
    plt.ylabel(ylabel_dict[theme])
    plt.suptitle('FY2015 EUAS {2} building data set with positive sq.ft.\n{0} (n = {1})'.format(filter_dict[theme], len(df), plotset))
    if theme == 'eui_water':
        plt.ylim((0, 60))
    else:
        plt.ylim((0, 140))
    P.savefig(os.getcwd() + '/plot_FY_annual/{1}_prog_num_{0}.png'.format(theme, plotset), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi), bbox_inches='tight')
    plt.close()

# temp usage
def join_cost():
    df1 = pd.read_csv(homedir + 'master_table/ecm_highlevelAction.csv', parse_dates=['Substantial Completion Date'])
    df1.replace(0, np.nan, inplace=True)
    # df2 = pd.read_csv(os.getcwd() + '/input/FY/ECM info/Light-Touch M_V - ARRA Targets to Actuals and Commissioning Details-sheet0.csv', parse_dates=['Substantial Completion Date'])
    df2 = pd.read_csv(os.getcwd() + '/input/FY/ECM info/Light-Touch M_V-sheet1.csv', parse_dates=['Substantial Completion Date'])
    df2 = df2[['Building ID', 'Substantial Completion Date', 'Total ARRA Obligation', 'Advanced Metering', 'Building Envelope', 'Building Tune Up', 'HVAC', 'Indoor Environmental Quality', 'Lighting']]
    df2.rename(columns={'Indoor Environmental Quality': 'IEQ', 'Building Tune Up': 'Building Tuneup or Utility Improvements'}, inplace=True)
    df2.replace('Y', 1, inplace=True)

    df2.dropna(subset=['Building ID', 'Substantial Completion Date'],
              inplace=True)
    print 'Drop na'
    print len(df2)
    df2_sum = df2[['Advanced Metering', 'Building Envelope', 
                 'Building Tuneup or Utility Improvements', 
                 'HVAC', 'IEQ', 'Lighting']]
    df2['num_ecm'] = df2_sum.sum(axis=1)
    df2 = df2[df2['num_ecm'] > 0]
    print 'has ecm'
    print len(df2)
    df2.drop_duplicates(cols=['Building ID', 'Advanced Metering',
                              'Building Envelope', 
                              'Building Tuneup or Utility Improvements',
                              'HVAC', 'IEQ', 'Lighting'], 
                       inplace=True)
    print 'drop dup'
    print len(df2)
    actions = ['Advanced Metering', 'Building Envelope', 'Building Tuneup or Utility Improvements', 'HVAC', 'IEQ', 'Lighting']
    df2.info()
    df = pd.merge(df1, df2, how='left', on=['Building ID'] + actions, suffixes=('_ecmsheet', '_costsheet'))
    # df = pd.merge(df1, df2, how='left', on='Building ID', suffixes=('_ecmsheet', '_costsheet'))
    df['Date Difference'] = (df['Substantial Completion Date_ecmsheet'] - df['Substantial Completion Date_costsheet']).abs()
    df['Date Difference_days'] = df['Date Difference'].map(lambda x: x / np.timedelta64(1, 'D'))
    print df['Date Difference_days'].describe()
    df.to_csv(homedir + 'master_table/ecm_cost_temp.csv', index=False)
    df.info()

def read_ecm_highlevel():
    print 'reading and cleaning high level ECM action files'
    df = pd.read_csv(os.getcwd() + \
                     '/input/FY/Portfolio HPGB Dashboard_highlevel.csv', parse_dates=['Substantial Completion Date'])
    df.to_csv(homedir + 'master_table/ECM/Portfolio HPGB Dashboard_highlevel.csv', index=False)
    ecm_cols = ['Substantial Completion Date', 'Building ID',
                'Advanced Metering', 'Building Envelope', 
                'Building Tuneup or Utility Improvements', 
                'HVAC', 'IEQ', 'Lighting']
    df = df[ecm_cols]
    print 'original, ', len(df)
    df.dropna(subset=['Building ID', 'Substantial Completion Date'],
              inplace=True)
    print 'Drop na, ', len(df)
    df_sum = df[['Advanced Metering', 'Building Envelope', 
                 'Building Tuneup or Utility Improvements', 
                 'HVAC', 'IEQ', 'Lighting']]
    df['num_ecm'] = df_sum.sum(axis=1)
    df = df[df['num_ecm'] > 0]
    print 'has ecm, ', len(df)
    df.drop_duplicates(cols=['Building ID', 'Advanced Metering',
                             'Building Envelope', 
                             'Building Tuneup or Utility Improvements',
                             'HVAC', 'IEQ', 'Lighting'], 
                       inplace=True)
    print 'drop dup, ', len(df)
    df = df[df['Substantial Completion Date'] < pd.Timestamp('2016-01-01')]
    print 'drop unfinished action, ', len(df)
    df.drop('num_ecm', axis=1, inplace=True)
    df.to_csv(homedir + 'master_table/ECM/Portfolio_HPGB_clean.csv', index=False)
    print 'number of building', len(set(df['Building ID'].tolist()))
    df2 = df.groupby('Building ID').max()
    # taking max of action and date
    df2.to_csv(homedir + 'master_table/ECM/Portfolio_HPGB_collapse.csv')
    df2 = pd.read_csv(homedir + 'master_table/ECM/Portfolio_HPGB_collapse.csv')
    df_gsadate = pd.read_csv(os.getcwd() + \
                             '/input/FY/GSAlink_Buildings_First_55_Opiton_26_Start_Stop_Dates.csv')
    df_gsadate.to_csv(homedir + 'master_table/ECM/GSAlink_Buildings_First_55_Opiton_26_Start_Stop_Dates.csv', index=False)
    df_gsadate = df_gsadate[['Building ID', 'Rollout Date']]
    print 'gsa date', len(df_gsadate)
    df_gsadate['GSALink'] = 1
    df_all = pd.merge(df2, df_gsadate, how='outer', on='Building ID')
    ecm_cols = list(df_all)
    ecm_cols.remove('Building ID')
    ecm_cols.remove('Substantial Completion Date')
    ecm_cols.remove('Rollout Date')
    total_action = df_all[ecm_cols].sum(axis=1)
    df_all['number of ECM action'] = total_action
    df_all['with ECM action'] = df_all['number of ECM action'].map(lambda x: 1 if x > 0 else 0)
    df_all.sort(columns=['Building ID', 'Substantial Completion Date',
                         'Rollout Date'], inplace=True)
    df_all.to_csv(homedir + 'master_table/ECM/ecm_highlevelAction.csv',
                  index=False)
    df_all_no_collapse = pd.merge(df, df_gsadate, how='outer', on='Building ID')
    df_all_no_collapse.to_csv(homedir +
                              'master_table/ECM/ecm_highlevelAction_no_collapse.csv',
                              index=False)

    conn = sqlite3.connect(homedir + 'db/ecm_info.db')
    c = conn.cursor()
    filelist = glob.glob(homedir + 'master_table/ECM/*.csv')
    for f in filelist:
        filename = f[(f.rfind('/') + 1): f.rfind('.')]
        filename = filename.replace(' ', '_')
        df = pd.read_csv(f)
        df.to_sql(filename, conn, if_exists='replace')
    conn.close()
    return
    
def read_icao():
    # source: http://weather.noaa.gov/tg/site.shtml
    with open (weatherdir + 'nsd_cccc.txt', 'r') as rd:
        lines = rd.readlines()
    print
    length = len(lines)
    for i in range(length):
        lines[i] = lines[i].replace(u"\u2018", "'")
        lines[i] = lines[i].replace(u"\u2019", "'")
        # manual correction:
        # 1. KSTZ columns not aligned to header
        # 2. KBAN, no country, in CA
        lines[i] = \
            lines[i].replace('KSTZ;--;---;South Timbalier;United States;4;28-09-35N;090-39-59W;;;;;;', 'KSTZ;--;---;;South Timbalier;United States;4;28-09-35N;090-39-59W;;;;;')
        lines[i] = \
            lines[i].replace('KBAN;--;---;MCMWTC BRIDGEPORT, CA;CA;;',
                             'KBAN;--;---;MCMWTC BRIDGEPORT, CA;CA;United States;')
    with open(weatherdir + 'weatherinput/nsd_cccc_format.txt', 'w+') as wt:
        wt.write(''.join(lines))
    names = ['ICAO Location Indicator', 'Block Number', 
             'Station Number', 'Place Name', 'State Abbr', 
             'Country Name', 'WMO Region', 'Station Latitude', 
             'Station Longitude', 'Upper Air Latitude', 
             'Upper Air Longitude', 'Station Elevation', 
             'Upper Air Elevation', 'RBSN indicator']
    df = pd.read_csv(weatherdir + 'weatherinput/nsd_cccc_format.txt', sep=';', header=None, names=names)
    # df.dropna(subset=['ICAO Location Indicator'], inplace=True)
    df.to_csv(weatherdir + 'weatherinput/nsd_cccc.csv', index=False)

def get_timezone(lat, lng, s):
    # timestamp is not important
    url = 'https://maps.googleapis.com/maps/api/timezone/json?location={0},{1}&timestamp={2}'.format(lat, lng, 1254355200)
    r = requests.get(url)
    if r.json()['status'] == 'ZERO_RESULTS':
        print '{0}, {1}, {2}, {3}'.format(s, lat, lng, np.nan)
        return np.nan
    else:
        result = r.json()['timeZoneId']
        print '{0}, {1}, {2}, {3}'.format(s, lat, lng, result)
        return result

def filter_icao():
    df2 = pd.read_csv(weatherdir + 'weatherinput/nsd_cccc_fill.csv')
    df2 = df2[['ICAO Location Indicator', 'Place Name', 'State Abbr',
               'Country Name', 'WMO Region', 'Station Latitude',
               'Station Longitude']]
    df2['Country Name'] = df2['Country Name'].map(lambda x: x.title())
    df2 = df2[df2['Country Name'] == 'United States']
    def get_latlon(s):
        tokens = s.split('-')
        if len(tokens) == 2:
            value = float(tokens[0]) + float(tokens[1][:-1])/60
            nsew = tokens[-1][-1]
        elif len(tokens) == 3:
            value = float(tokens[0]) + float(tokens[1])/60 + \
                    float(tokens[2][:-1])/3600
            nsew = tokens[-1][-1]
        if nsew == 'N' or nsew == 'E':
            return value
        else:
            return (-1.0) * value
    df2['Lat'] = df2['Station Latitude'].map(lambda x: get_latlon(x))
    df2['Long'] = df2['Station Longitude'].map(lambda x: get_latlon(x))
    timezone_input = zip(df2['Lat'].tolist(), df2['Long'].tolist(), df2['ICAO Location Indicator'].tolist())
    step = 10
    for j in range(len(timezone_input)/step):
        for i in range(j * step, (j + 1) * step):
            get_timezone(timezone_input[i][0], timezone_input[i][1],
                         timezone_input[i][2])
        time.sleep(1)
    df2.to_csv(weatherdir + 'weatherinput/nsd_cccc_us.csv',
               index=False)
    print len(df2)

def getICAO_geocoder(StateAbbr, Address, zipcode, buildingId,
                     df_lookup):
    # print StateAbbr, Address, zipcode
    if type(Address) == float and np.isnan(Address):
        print '{0},{1},{2},{3},{4}'.format(buildingId, np.nan, np.nan,
                                           np.nan, np.nan)
        return (np.nan, np.nan)
    df = df_lookup.copy()
    g = geocoder.google('{0},{1}'.format(Address, StateAbbr))
    if not (g.json['ok']):
        print '{0},{1},{2},{3},{4}'.format(buildingId, 'Not Found', -1.0, np.nan, np.nan)
        return ('Not Found', -1.0)
    df['distance'] = df.apply(lambda r: vincenty(g.latlng, (r['Lat'], r['Long']), miles=True), axis=1)
    min_distance = df['distance'].min()
    df = df[df['distance'] == min_distance]
    print '{0},{1},{2},{3},{4}'.format(buildingId, 
                               # df['ICAO Location Indicator'].iloc[0], 
                               df['ICAO'].iloc[0], 
                               min_distance, g.latlng[0], g.latlng[1])
    # return (df['ICAO Location Indicator'].iloc[0], min_distance)
    return (df['ICAO'].iloc[0], min_distance)

def getICAO(StateAbbr, Address, zipcode, df_lookup):
    print StateAbbr, Address, zipcode
    if type(Address) == float and np.isnan(Address):
        return (np.nan, np.nan)
    df = df_lookup.copy()
    counter = 0
    geolocator = Nominatim()
    location = geolocator.geocode('{0},{1}'.format(Address, StateAbbr))
    print
    # print location
    #location = geolocator.geocode('{0}'.format(zipcode))
    if location == None:
        return ('Not Found', -1.0)
    print location.latitude, location.longitude
    df['distance'] = df.apply(lambda r: vincenty((location.latitude, location.longitude), (r['Lat'], r['Long'])).miles, axis=1)
    min_distance = df['distance'].min()
    print min_distance
    df = df[df['distance'] == min_distance]
    print (df['ICAO Location Indicator'].iloc[0], min_distance)
    return (df['ICAO Location Indicator'].iloc[0], min_distance)

def match_station():
    df_building = pd.read_csv(homedir + 'master_table/static_info.csv')
    df_building = df_building[['Building Number', 'Street', 'State Abbr', 'Zip Code']]
    # df_stationlookup = pd.read_csv(weatherdir + \
    #                                'weatherinput/nsd_cccc_us.csv')
    df_stationlookup = pd.read_csv(weatherdir + \
                                   'weatherinput/Weather Station Mapping.csv')

    step = 10
    dfs = [df_building[i * step: (i + 1) * step] for i in range(0, len(df_building)/step + 1)]
    # dfs = [df_building]
    # modify starting point to resume if geocoding return an error
    for i in range(134, len(dfs)):
        # print i, len(dfs[i])
        # print dfs[i].head()
        dfs[i]['Weather Station and dist'] = \
            dfs[i].apply(lambda r: \
                getICAO_geocoder(r['State Abbr'], r['Street'], 
                                 r['Zip Code'], r['Building Number'],
                                 df_stationlookup), axis=1)
        # dfs[i]['ICAO'] = dfs[i]['Weather Station and dist'].map(lambda x: x if type(x) == float else x[0])
        # dfs[i]['distance [mile]'] = dfs[i]['Weather Station and dist'].map(lambda x: x if type(x) == float else x[1])
        # dfs[i].drop('Weather Station and dist', axis=1, inplace=True)
        # # dfs[i].to_csv(homedir + 'master_table/weather_info/location_info_{0}.csv'.format(i), index=False)
        # dfs[i].to_csv(homedir + 'master_table/weather_info_geocoder/location_info_{0}.csv'.format(i), index=False)
        time.sleep(1)

def join_timezone():
    df_s = pd.read_csv(weatherdir + 'weatherinput/nsd_cccc_us.csv')
    names = ['ICAO Location Indicator', 'Lat', 'Long', 'timeZoneId']
    df_t = pd.read_csv(weatherdir + 'weatherinput/timezone.txt',
                       header=None, names=names)
    df_t = df_t[['ICAO Location Indicator', 'timeZoneId']]
    print df_t['timeZoneId'].value_counts()
    df_t.replace('nan', np.nan, inplace=True)
    print df_t['timeZoneId'].value_counts()
    df_all = pd.merge(df_s, df_t, how='left', 
                      on='ICAO Location Indicator')
    df_all.to_csv(weatherdir + 'weatherinput/nsd_cccc_us_tz.csv',
                  index=False)

# run on the next day
def re_geocode():
    df_building = pd.read_csv(homedir + 'master_table/highDist.csv')
    df_building = df_building[['Building Number', 'Street', 'State Abbr', 'Zip Code']]
    df_stationlookup = pd.read_csv(weatherdir + \
                                   'weatherinput/nsd_cccc_us.csv')
    step = 10
    dfs = [df_building[i * step: (i + 1) * step] for i in range(0, len(df_building)/step)]
    for i in range(0, len(dfs)):
        print i, len(dfs[i])
        starttime = time.time()
        dfs[i]['Weather Station and dist'] = dfs[i].apply(lambda r: getICAO_geocoder(r['State Abbr'], r['Street'], r['Zip Code'], df_stationlookup), axis=1)
        dfs[i]['ICAO'] = dfs[i]['Weather Station and dist'].map(lambda x: x if type(x) == float else x[0])
        dfs[i]['distance [mile]'] = dfs[i]['Weather Station and dist'].map(lambda x: x if type(x) == float else x[1])
        endtime = time.time()
        dfs[i].drop('Weather Station and dist', axis=1, inplace=True)
        dfs[i].to_csv(homedir + 'master_table/weather_info_geocoder_highDist/location_info_{0}.csv'.format(i), index=False)
    print endtime - starttime
    print 'end'

def concat_weather_station():
    filelist = glob.glob(homedir + '/master_table/weather_info_geocoder_step10/*.csv')
    dfs = [pd.read_csv(f) for f in filelist]
    df = pd.concat(dfs, ignore_index=True)
    df.sort(columns=['Building Number'], inplace=True)
    print df.describe()
    df.to_csv(homedir + '/master_table/building_station.csv',
              index=False)

# adapted from Shilpi's code
def get_weather_data(s, minDate, maxDate):
    print 'start reading {0}'.format(s)
    starttime = time.time()
    # FIXME: update this
    url =  "https://128.2.109.159/piwebapi/dataservers/s0-MYhSMORGkyGTe9bdohw0AV0lOLTYyTlBVMkJWTDIw/points?namefilter=*underground/*"+s+"*tempe*"
    r = requests.get(url, auth=('Weather', 'Weather1!@'), verify=False)
    if len(r.json()['Items']) == 0:
        print 'No Data for station {0}'.format(s)
        return
    webId = r.json()['Items'][0]['WebId']
    recordUrl = "https://128.2.109.159/piwebapi/streams/"+webId+"/interpolated?starttime='"+minDate+"'&endtime='"+maxDate+"'&maxcount=149000"
    rec = requests.get(recordUrl, auth=('Weather', 'Weather1!@'),
                       verify=False)
    json_list = (rec.json()['Items'])
    timestamps = [x['Timestamp'] for x in json_list]
    temp = [x['Value'] for x in json_list]
    df = pd.DataFrame({'Timestamp': timestamps, s: temp})
    df['localTime'] = pd.date_range(minDate, periods=len(df), freq='H')
    df.to_csv(weatherdir + \
              'weatherinput/by_station/{0}.csv'.format(s), index=False)
    endtime = time.time()
    print 'finish reading {0} in {1}s'.format(s, endtime - starttime)

def join_station():
    df_static = pd.read_csv(homedir + 'master_table/static_info.csv')
    df_bd_station = pd.read_csv(weatherdir + \
                                'weatherinput/geocoding_log.txt')
    df_bd_station.to_csv(homedir + 'master_table/building_station.csv')
    df_bd_station['Lat_building'] = df_bd_station.apply(lambda r: r['Lat_building'] if r['ICAO'] != 'Not Found' else 'Not Found', axis=1)
    df_bd_station['Long_building'] = df_bd_station.apply(lambda r: r['Long_building'] if r['ICAO'] != 'Not Found' else 'Not Found', axis=1)
    df_static_station = pd.merge(df_static, df_bd_station, how='left',
                                 on='Building Number')
    df_static_station.to_csv(homedir + \
                             'master_table/static_info_ws.csv',
                             index=False)

    # df_t = pd.read_csv(weatherdir + 'weatherinput/nsd_cccc_us_tz.csv')
    # df_all = pd.merge(df_static_station, df_t, how='left', on='ICAO')
    # df_all.sort(columns=['Building Number'], inplace=True)
    # df_all.to_csv(homedir + 'master_table/static_info_fullws.csv',
    #               index=False)
    print 'end join_station()'

# join stations retrieved from gsalink, deprecated. from joining, all
# addresses are included in the static table for gsalink buildings
def join_station_gsa_dep():
    df1 = pd.read_csv(homedir + 'master_table/static_info_ws.csv')
    df1.info()
    df2 = pd.read_csv(weatherdir + 'gsa_station.csv')
    df2 = df2[['Building ID', 'Street', 'ICAO', 'distance [mile]']]
    df2.rename(columns={'Building ID': 'Building Number'},
               inplace=True)
    df = pd.merge(df1, df2, how='left', on='Building Number',
                  suffixes=('', '_gsa'))
    df.info()
    df.to_csv(homedir + 'master_table/static_info_ws_gsa.csv',
              index=False)

def join_mapped_station():
    filelist = glob.glob(weatherdir + 'weatherinput/by_station/*.csv')
    stations = [f[-8: -4] for f in filelist]
    df = pd.DataFrame({'ICAO': stations, 
                       'Download Weather Data': [1] * len(stations)})
    df_weather = pd.read_csv(homedir + 'master_table/static_info_ws.csv')
    df_weather = df_weather[['Building Number', 'Street', 'City', 
                             'Zip Code', 'ICAO', 'distance [mile]', 
                             'Lat_building', 'Long_building']]
    df_err_cnt = pd.read_csv(weatherdir + 'weatherinput/station_errline_count.csv')
    df_err_cnt = df_err_cnt[['ICAO', 'count']]
    df_err_cnt.rename(columns={'count': 'missing_hour_count'},
                      inplace=True)
    df_all = pd.merge(df_weather, df, how='left', on = 'ICAO')
    df_all2 = pd.merge(df_all, df_err_cnt, how='left', on='ICAO')
    df_all2['Valid Weather Data'] = df_all2.apply(lambda r: 1 if r['Download Weather Data'] == 1 and not r['missing_hour_count'] > 20 else np.nan, axis=1)
    df_all2.sort(columns=['Building Number'], inplace=True)
    df_all2.to_csv(homedir + \
                   'master_table/building_stationAvailability.csv',
                   index=False)
    df_all2.info()
    print 'end join_mapped_station'

def need_to_read():
    df1 = pd.read_csv(homedir + 'master_table/static_info_ws.csv')
    need = set(df1['ICAO'].tolist())
    filelist = glob.glob(weatherdir + 'weatherinput/by_station/*.csv')
    have = set([f[-8: -4] for f in filelist])
    read = list(need.difference(have))
    read.remove(np.nan)
    read.remove('Not Found')
    return read

def get_all_station_loc():
    # read_icao()
    # filter_icao()
    # match_station()
    # join_timezone()
    # join_station()
    # read_weather_data(need_to_read())
    # join_mapped_station()
    return

def read_weather_data(read):
    if read == None:
        df_station = pd.read_csv(homedir + \
                                'master_table/building_station.csv')
        stations = list(set(df_station['ICAO'].tolist()))
    else:
        stations = read
    minDate = '2007-10-01 00:00:00'
    maxDate = '2016-01-01 00:00:00'

    length = len(stations)
    for i in range(0, length):
        s = stations[i]
        if type(s) == float:
            continue
        get_weather_data(s, minDate, maxDate)
    return

def join_static_detailECM():
    print 'join static_info.csv to detail ECM ...'
    df1 = pd.read_csv(os.getcwd() + '/input/FY/ScopePortfolioReport_20160105-5.csv', header=range(3))
    df1.columns = ['_'.join(col).strip() for col in df1.columns.values]
    subset = list(df1)[2:]
    df1.dropna(subset=subset, inplace=True, how='all')
    df11 = df1.groupby('Building ID_Building ID_Building ID').max()
    df11.to_csv(homedir + 'master_table/ecm_detail_merge.csv')
    df2 = pd.read_csv(homedir + 'master_table/static_info.csv')
    df = pd.merge(df2, df11, left_on='Building Number', right_on=df11.index, how='left')
    df.sort('Building Number', inplace=True)
    df.to_csv(homedir + 'master_table/static_info_detailECM.csv',
              index=False)
    df3 = df2[['Building Number', 'Cat_latest']]
    df_all = pd.merge(df3, df11, left_on='Building Number',
                      right_on=df11.index)
    df_all.sort('Building Number', inplace=True)
    df_all.to_csv(homedir + 'master_table/euasset_detailECM.csv',
                  index=False)

def process_master(yearlist):
    print 'processing master tables ...'
    # sanity_check_static_long(yearlist)
    # energy_info(yearlist)
    # join_static()
    # read_ecm_highlevel()
    # join_static_ecm()
    # join_indicator_static_all(['has_data >= 8 years', 
    #                            'good_both >= 8 years', 'good_area_15'])
    # join_static_detailECM()
    # get_all_station_loc()
    # concat_weather_station()
    # re_geocode()
    # BOOKMARK: drop cost
    # join_cost()
    return

# if file extension is .xls, it cannot be properly converted, need to 
def excel2csv_singlesheet(yearlist):
    df = pd.read_csv(homedir + 'master_table/energy_info.csv')
    for year in yearlist:
        print 'converting EUAS {0} data to csv ...'.format(year)
        filelist = \
            glob.glob(os.getcwd() + \
                      '/input/FY/EUAS/EUAS {0}/*.xls'.format(year))
        if len(filelist) > 0:
            print '.xls files not processed'
            print [f[f.rfind('/') + 1:] for f in filelist]
    for year in yearlist:
        filelist = \
            glob.glob(os.getcwd() + \
                      '/input/FY/EUAS/EUAS {0}/*.xlsx'.format(year))
        for f in filelist:
            filename = f[f.rfind('/') + 1:]
            print 'reading file {0}'.format(filename)
            tokens = filename.split(' ')
            df = pd.read_excel(f, sheetname=0)
            df.to_csv('{0}sep/FY{1}_{2}.csv'.format(homedir,
                                                    str(year)[2:],
                                                    tokens[1]),
                      index=False)

def input_energy(cutoffyear):
    # yearlist = [2003, 2004, 2007, 2008, 2009, 2016]
    yearlist = range(2003, 2017)
    agg_yearlist = range(2003, 2017)
    agg_cal_yearlist = range(2002, 2016)

    # excel2csv()
    # excel2csv_singlesheet(yearlist)
    # # building_info() # deprecated
    # fiscal2calendar()
    # region2building()
    # region2building_cal()
    # calculate('fis')
    # calculate('cal')
    # # the FY data currently available
    # aggregate_allyear('fis', agg_yearlist)
    # aggregate_allyear('cal', agg_cal_yearlist)
    # get_raw_concat(agg_yearlist, 'fis')
    # get_raw_concat(agg_cal_yearlist, 'cal')
    # get_flow_reorg(agg_yearlist, 'fis', cutoffyear)
    # get_flow_reorg(agg_cal_yearlist, 'cal', cutoffyear)
    # for theme in ['has_data', 'good_elec', 'good_gas', 'good_water',
    #               'good_both']:
    #     plot_building_data_repl(agg_yearlist, theme, 'fis', 'All', cutoffyear)
    #     plot_building_data_repl(agg_yearlist, theme, 'fis', 'AI', cutoffyear)
    input_to_db()
    return

def input_to_db():
    conn = sqlite3.connect(homedir + 'db/eui_fiscal.db')
    c = conn.cursor()
    print 'dumping to eui_info'
    filelist = glob.glob(homedir + 'filter_bit/fis/eui_all_*.csv')
    for f in filelist:
        filename = f[f.rfind('/') + 1: -4]
        filename = filename.replace('all_', 'FY')
        print filename
        pd.read_csv(f).to_sql(filename, conn, if_exists='replace')
    conn.close()
    return

def main():
    yearlist = [2005, 2006]
    agg_yearlist = range(2003, 2017)
    agg_cal_yearlist = range(2002, 2016)
    # input_energy(8)
    # plot_action_by_prog()
    process_master(agg_yearlist)
    # plot_static('office')
    # plot_static('All', 'has_data >= 8 years', 'good_both >= 8 years', 'has_data >= 8 years', 'has at least 8 year EUAS data', theme_dict['good_both'], 'has at least 8 year EUAS data')
    # join_station_gsa()
    # plot_program('All', 'FY2015')
    # plot_program_energy('eui', 'all')
    # plot_program_energy('all')
    # for theme in ['eui', 'eui_elec', 'eui_gas', 'eui_water']:
    #     # plot_program_yesno(theme, 'office')
    #     # plot_program_yesno(theme, 'all')
    #     plot_program_num(theme, 'office')
    #     plot_program_num(theme, 'all')

    # get_fuel_type(agg_yearlist)
    # join_fueltype(agg_yearlist)
    # fuel_type_plot(agg_yearlist, 'AI')
    # fuel_type_plot(agg_yearlist, 'All')

    #euas2csv()
    #join_program()
    #report_false()
    #report_false_15()
    #weather_dict('none')
    #check_use_dupe()
    return

main()
