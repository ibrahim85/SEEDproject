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
def get_raw_concat():
    year_col = 'Fiscal Year'
    month_col = 'Fiscal Month'
    pre = 'FY'
    yearlist = range(2010, 2016)

    for year in yearlist:
        label = str(int(year))[-2:]
        filelist = ['{0}/csv_FY/sep/FY{1}_{2}.csv'.format(os.getcwd(), label, i) for i in range(1, 12)]
        print filelist
        dfs = [pd.read_csv(f) for f in filelist]
        df_all = pd.concat(dfs, join='inner', ignore_index=True)
        df_all.to_csv(os.getcwd() + \
                '/csv_FY/raw_concat/{0}{1}.csv'.format(pre, label),
                index=False)

def sanity_check_static():
    yearlist = range(2010, 2016)
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
        print label
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
    changing_summary = []
    for col in rename_cols:
        cols = [c for c in list(df_all) if col in c]
        print cols
        c = '{0}_is_changing'.format(col)
        df_all[c] = df_all.apply(lambda r: is_changing(r, cols),
                                 axis=1)
        changing_summary.append(c)
    newcols = ['Building Number'] + changing_summary + \
        reduce(lambda x, y: x + y, [[suf(c, l) for l in labellist] for
                                    c in rename_cols])
    print newcols
    df_all = df_all[newcols]
    df_all.sort(columns=['Building Number'], inplace=True)
    df_all.to_csv(os.getcwd() + '/csv_FY/master_table/static_info_EUAS_check.csv', index=False)
    def get_value(r, cols):
        values = [r[col] for col in cols]
        values = list(set(values))
        values = [v for v in values if not (type(v) == float and
                                            np.isnan(v))]
        return values[0]
    drop_cols = []
    df_all.drop(['Region No._is_changing', 'State_is_changing'],
                axis=1, inplace=True)
    for col in ['Region No.', 'State']:
        cols = [c for c in list(df_all) if col in c]
        df_all[col] = df_all.apply(lambda r: get_value(r, cols),
                                   axis=1)
        drop_cols += cols
    df_all.drop(drop_cols, axis=1, inplace=True)
    df_all.to_csv(os.getcwd() + '/csv_FY/master_table/static_info_EUAS.csv', index=False)

def add_filter_bit():
    calOrFiscal = 'fis'
    office_set = get_office()
    yearlist = range(2010, 2016)
    for yr in yearlist:
        label = str(int(yr))[-2:]
        df_zero = pd.read_csv(os.getcwd() + '/csv_FY/raw_concat/FY{0}.csv'.format(label))
        df_zero.drop_duplicates(cols='Building Number', inplace=True)
        df_zero.to_csv(os.getcwd() + '/csv_FY/raw_concat/FY{0}_drop.csv'.format(label), index=False)
        df_zero = df_zero[['Building Number']]
        df = pd.read_csv(os.getcwd() + '/csv_FY/agg/eui_{0}.csv'.format(yr))
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
        df_all.drop(['eui_elec', 'eui_gas', 'eui_oil', 'eui_water', 'eui', 'Region No.', 'Fiscal Year', 'Cat', 'eui_steam'], axis = 1, inplace=True)
        df_all.to_csv(os.getcwd() +
                      '/csv_FY/filter_bit/{1}/eui_clean_20{0}.csv'.format(label, calOrFiscal), index=False)
        df.to_csv(os.getcwd() +
                  '/csv_FY/filter_bit/{1}/eui_20{0}.csv'.format(label,
                                                                calOrFiscal), index=False)

# merge quality indicator files
def merge_indicator():
    calOrFiscal = 'fis'
    filelist = glob.glob(os.getcwd() + '/csv_FY/filter_bit/{0}/eui_clean_*'.format(calOrFiscal))
    dfs = [pd.read_csv(csv) for csv in filelist]
    all_building = reduce(set.union, [get_building(df) for df in dfs])
    df_base = pd.DataFrame({'Building Number': list(all_building)})
    df_list = [df_base] + dfs
    df_all = reduce(lambda x, y: pd.merge(x, y, how='left', on='Building Number'), df_list)
    df_all.to_csv(os.getcwd() + '/csv_FY/filter_bit/{0}/indicator.csv'.format(calOrFiscal), index=False)
    return

# report number of each filter
def report_number():
    calOrFiscal = 'fis'
    def suf(title, yr):
        return '{0}_{1}'.format(title, str(int(yr))[-2:])
    # for separate year:
    print 'report all building'
    df = pd.read_csv(os.getcwd() + '/csv_FY/filter_bit/{0}/indicator.csv'.format(calOrFiscal))
    if calOrFiscal == 'cal':
        yearlist = range(2009, 2016)
    else:
        yearlist = range(2010, 2016)

    for yr in yearlist:
        label = str(int(yr))[-2:]
        themes = [suf(x, yr) for x in ['has_data', 'good_area', 'good_elec', 'good_gas', 'good_water', 'good_both', 'good_all']]
        for theme in themes:
            print theme
            print df[theme].sum()
        print 'office'
        for theme in themes:
            print 'office: {0}'.format(theme)
            df_temp = df[df[suf('office', yr)] == 1]
            print df_temp[theme].sum()
    themes = ['has_data', 'good_area', 'good_elec', 'good_gas', 'good_water', 'good_both', 'good_all', 'office']
    for theme in themes:
        sumcols = ['{0}_{1}'.format(theme, str(int(yr))[-2:]) \
                for yr in yearlist]
        print sumcols
        df['{0}_sum'.format(theme)] = df[sumcols].sum(axis=1)
        total_years = len(yearlist)
        df[theme] = df['{0}_sum'.format(theme)].map(lambda x: 1 if x == total_years else 0)
        #df[theme] = df.apply(lambda row: 1 if row[theme + '_13'] + row[theme + '_14'] + row[theme + '_15'] == 3 else 0, axis=1)
        print theme
        print df[theme].sum()
    print 'office'
    df.to_csv(os.getcwd() + '/csv_FY/filter_bit/{0}/indicator_all.csv'.format(calOrFiscal), index=False)
    df_temp = df[df['office'] == 1]
    for theme in themes:
        print 'office: {0}'.format(theme)
        print df_temp[theme].sum()

# process starts from positive floor area
def get_flow_reorg():
    add_filter_bit()
    merge_indicator()
    report_number()

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
    filelist = glob.glob(os.getcwd() + '/csv_FY/sep/*.csv')
    for csv in filelist:
        df = pd.read_csv(csv)
        year = int(df.ix[0, 'Fiscal Year'])
        bds = set(df['Building Number'].tolist())
        for b in bds:
            df_b = df[df['Building Number'] == b]
            outfile = (os.getcwd() + \
                    '/csv_FY/single/{0}_{1}.csv'.format(b, year))
            df_b.to_csv(outfile, index=False)

# region sheet to single building files, in calendar year
def region2building_cal():
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

# BOOKMARK : region_year files
# output FY{year}_{region}.csv files
def fiscal2calendar():
    for i in range(1, 12):
        filelist = glob.glob(os.getcwd() + '/csv_FY/sep/*_{0}*.csv'.format(i))
        dfs = [pd.read_csv(csv) for csv in filelist]
        df_con = pd.concat(dfs, ignore_index=True)
        df_con['month'] = df_con['Fiscal Month'].map(fiscal2calmonth)
        df_con['year'] = df_con.apply(lambda row: fiscal2calyear(row['Fiscal Year'], row['Fiscal Month']), axis=1)
        print df_con[['Fiscal Year', 'year', 'Fiscal Month', 'month']].head()
        gr = df_con.groupby('year')
        for name, group in gr:
            yr = str(int(name))[-2:]
            rg = i
            outfile = os.getcwd() + '/csv_FY/cal/FY{0}_{1}.csv'.format(yr, rg)
            print outfile
            group.sort(columns=['Building Number', 'month'],
                       inplace=True)
            group.to_csv(outfile, index=False)

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
    # remove later
    #filelist = [f for f in filelist if 'TX0000LW' in f]

    for csv in filelist:
        df = pd.read_csv(csv)
        filename = csv[csv.find(identifier) + len(identifier) + 1:]
        print filename
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
        bd = df.ix[0, 'Building Number']
        yr = int(df.ix[0, year_col])
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

def aggregate(year, calOrFiscal):
    if calOrFiscal == 'cal':
        in_folder = 'single_eui_cal'
        out_folder = 'agg_cal'
        year_col = 'year'
        month_col = 'month'
    else:
        in_folder = 'single_eui'
        out_folder = 'agg'
        year_col = 'Fiscal Year'
        month_col = 'Fiscal Month'
    filelist = glob.glob(os.getcwd() +
                         '/csv_FY/{1}/*{0}.csv'.format(year, in_folder))
    dfs = []
    for csv in filelist:
        df = pd.read_csv(csv)
        filename = csv[(csv.find(in_folder) + len(in_folder) + 1):]
        # check monthly records availability
        '''
        if (len(df) != 12 or len(df['Fiscal Month'].unique()) != 12):
            print filename
        '''
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
    print list(df_yr)
    df_yr = df_yr.sort(columns='Region No.')
    df_yr.to_csv(os.getcwd() + \
            '/csv_FY/{1}/eui_{0}.csv'.format(year, out_folder), index=False)

# fix me
def aggregate_allyear(calOrFiscal):
    if calOrFiscal == 'cal':
        yearlist = range(2009, 2016)
        for year in yearlist:
            aggregate(year, calOrFiscal)
    else:
        yearlist = range(2010, 2016)
        for year in yearlist:
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
    for y in years:
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
        df_sum.info()
        for col in ['Gas + Oil', 'Gas + Steam', 'Oil + Steam', 
                    'Gas + Oil + Steam', 'Gas Only', 'Oil Only', 
                    'Steam Only']:
            df_sum[col] = df_sum[col].apply(lambda x: 1 if x else 0)
        df_sum.info()
        df_sum.to_csv(os.getcwd() +
                      '/csv_FY/fuel_type/FY{0}.csv'.format(y))

def fuel_type_plot():
    yearlist = range(10, 16)
    filelist = ['{0}fuel_type/FY{1}.csv'.format(homedir, yr) for yr in yearlist]
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
    fuel_type_cols = ['None (all electric?)', 'Gas Only', 'Oil Only', 
                      'Steam Only', 'Gas + Oil', 'Gas + Steam', 
                      'Oil + Steam', 'Gas + Oil + Steam']
    df_all['Heating Fuel Type'] = \
        df_all.apply(lambda r: select_col(r, fuel_type_cols), axis=1)
    print df_all['Heating Fuel Type'].value_counts()
    df_all = df_all[['Heating Fuel Type', 'year']]
    sns.set_style("white")
    sns.set_palette(sns.color_palette('Set2'))
    sns.set_context("talk", font_scale=1.0)
    sns.mpl.rc("figure", figsize=(10, 5))
    sns.countplot(x='year', order= [str(x) for x in range(2010, 2016)],
                  hue='Heating Fuel Type', palette='Set3',
                  hue_order=fuel_type_cols, data=df_all)
    plt.legend(loc = 2, bbox_to_anchor=(1, 1))
    my_dpi=300
    plt.title('Heating Fuel Type Count (FY 2010, 2012-2015)')
    plt.ylabel('Number of Buildings')
    plt.xlabel('Fiscal Year')
    P.savefig(os.getcwd() + '/plot_FY_annual/fuel_type.png', dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi), bbox_inches='tight')
    plt.close()
    
# join fuel types to filter bit
def join_fueltype():
    indicator_df = pd.read_csv(homedir + 'filter_bit/fis/indicator_all.csv')

    yearlist = range(10, 16)
    cols = ['None (all electric?)', 'Gas Only', 'Oil Only', 
            'Steam Only', 'Gas + Oil', 'Gas + Steam', 
            'Oil + Steam', 'Gas + Oil + Steam', 'Chilled Water']
    dfs = []
    for yr in yearlist:
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
    return

def concat_all():
    filelist = [homedir + 'raw_concat/FY{0}.csv'.format(x) for x in
                range(10, 16)]
    dfs = []
    # take the average of No.1 to 6
    m_oil = (139 + 138 + 146 + 150)/4
    for f in filelist:
        filename = f[f.rfind('/') + 1:]
        print filename
        df = pd.read_csv(f)
        df = df[['Building Number', 'Fiscal Year', 'Fiscal Month',
                 'Electricity (KWH)', 'Steam (Thou. lbs)', 
                 'Gas (Cubic Ft)', 'Oil (Gallon)', 
                 'Chilled Water (Ton Hr)', 'Water (Gallon)']]
        df['Electricity (kBtu)'] = df['Electricity (KWH)'] * 3.412
        df['Gas (kBtu)'] = df['Gas (Cubic Ft)'] * 1.026
        df['Oil (kBtu)'] = df['Oil (Gallon)'] * m_oil
        df['Steam (kBtu)'] = df['Steam (Thou. lbs)'] * 1194
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.sort(columns=['Building Number', 'Fiscal Year', 
                         'Fiscal Month'], inplace=True)
    df_all.to_csv(homedir + 'master_table/energy_info.csv', index=False)

def join_static():
    df1 = pd.read_csv(homedir + 'master_table/static_info_EUAS.csv')
    df2 = pd.read_csv(os.getcwd() + '/input/FY/static info/Entire GSA Building Portfolio.csv')
    df2 = df2[['Building ID', 'Street', 'City', 'State', 'Zip Code',
               'Gross Square Feet (GSF)', 'Building Name', 
               'Building Class', 'Predominant Use', 
               'Owned or Leased Indicator']]
    df2.rename(columns={'State': 'State Abbr', 
                        'Building ID': 'Building Number'}, 
               inplace=True)
    df_all = pd.merge(df1, df2, how='left', on='Building Number')
    df_all.to_csv(homedir + 'master_table/static_info.csv',
                  index=False)
    print 'Finish join_static()'

# temp usage
def join_cost():
    df1 = pd.read_csv(homedir + 'master_table/ecm_highlevel.csv', parse_dates=['Substantial Completion Date'])
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
    df = pd.read_csv(os.getcwd() + \
                     '/input/FY/Portfolio HPGB Dashboard_highlevel.csv')
    ecm_cols = ['Substantial Completion Date', 'Building ID',
                'Advanced Metering', 'Building Envelope', 
                'Building Tuneup or Utility Improvements', 
                'HVAC', 'IEQ', 'Lighting']
    df = df[ecm_cols]
    print 'original'
    print len(df)
    df.dropna(subset=['Building ID', 'Substantial Completion Date'],
              inplace=True)
    print 'Drop na'
    print len(df)
    df_sum = df[['Advanced Metering', 'Building Envelope', 
                 'Building Tuneup or Utility Improvements', 
                 'HVAC', 'IEQ', 'Lighting']]
    df['num_ecm'] = df_sum.sum(axis=1)
    df = df[df['num_ecm'] > 0]
    print 'has ecm'
    print len(df)
    df.drop_duplicates(cols=['Building ID', 'Advanced Metering',
                             'Building Envelope', 
                             'Building Tuneup or Utility Improvements',
                             'HVAC', 'IEQ', 'Lighting'], 
                       inplace=True)
    print 'drop dup'
    print len(df)
    df_gsalink = pd.read_csv(os.getcwd() + '/input/FY/GSAlink 81 Buildings Updated 9_22_15.csv')
    df_gsalink.info()
    df_gsadate = pd.read_csv(os.getcwd() + '/input/FY/GSAlink_Buildings_First_55_Opiton_26_Start_Stop_Dates.csv')
    df_gsadate = df_gsadate[['Building ID', 'Rollout Date']]
    print 'gsa date', len(df_gsadate)
    df_gsadate['GSALink'] = 1
    df_all = pd.merge(df, df_gsadate, how='outer', on='Building ID')
    print 'all', len(df_all)
    df_all.drop('num_ecm', axis=1, inplace=True)
    df_all.sort(columns=['Building ID', 'Substantial Completion Date',
                         'Rollout Date'], inplace=True)
    print 'num of buildings', len(set(df_all['Building ID'].tolist()))
    df_all.to_csv(homedir + 'master_table/ecm_highlevel.csv',
                  index=False)
    
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
    join_mapped_station()
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

def process_master():
    # sanity_check_static()
    # concat_all()
    # join_static()
    # read_ecm_highlevel()
    # get_all_station_loc()
    # concat_weather_station()
    # re_geocode()
    # BOOKMARK: drop cost
    join_cost()
    return

def main():
    process_master()
    #excel2csv()
    #building_info() # deprecated
    #fiscal2calendar()
    #region2building()
    #region2building_cal()
    #calculate('fis')
    #calculate('cal')
    #aggregate_allyear('fis')
    #aggregate_allyear('cal')
    #get_raw_concat()
    # get_flow_reorg()

    #get_fuel_type(range(10, 16))
    #join_fueltype()
    #fuel_type_plot()

    #euas2csv()
    #join_program()
    #report_false()
    #report_false_15()
    #weather_dict('none')
    #check_use_dupe()
    return
main()
