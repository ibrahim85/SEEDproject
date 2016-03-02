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
from geopy.distance import vincenty
import time.time()

homedir = os.getcwd() + '/csv_FY/weather/'

# label for plotting
ylabel_dict = {'eui':'Electricity + Gas [kBtu/sq.ft]',
               'eui_elec':'Monthly Electricity [kBtu/sq.ft]',
               'eui_gas':'Monthly Natural Gas [kBtu/sq.ft]',
               'eui_oil':'Oil [Gallons/sq.ft]',
               'all': 'Electricity-Gas [kBtu/sq.ft]',
               'eui_water':'Water [Gallons/sq.ft]'}

title_dict = {'eui':'Electricity + Gas',
              'eui_elec':'Electricity',
              'eui_gas':'Gas',
              'eui_oil':'Oil',
              'all': 'Combined Electricity and Gas',
              'eui_water':'Water'}

kind_dict = {'temp': 'Temperature', 'hdd': 'HDD', 'cdd': 'CDD', 'all': 'Combined'}

# plot title
title_weather = {'eui':'Original and Weather Normalized '\
                      'Electricity + Gas Consumption',
                'eui_elec':'Original and Weather Normalized '
                           'Electricity Consumption',
                'eui_gas':'Original and Weather Normalized Natural '\
                          'Gas Consumption',
                'eui_oil':'Original and Weather Normalized Oil Consumption',
                'eui_water':'Original and Weather Normalized '\
                            'Water Consumption'}

title_dict_3 = {'eui':'Weather Normalized Electricity + Gas Consumption', 'eui_elec':'Weather Normalized Electricity Consumption', 'eui_gas':'Weather Normalized Natural Gas Consumption', 'eui_oil':'Weather Normalized Oil Consumption', 'eui_water':'Weather Normalized Water Consumption'}

xlabel_dict = {'temp': 'Monthly Mean Temperature, Deg F',
               'hdd': 'Monthly HDD, Deg F',
               'all': 'Monthly HDD(-)/CDD(+), Deg F',
               'cdd': 'Monthly CDD, Deg F'}

# read weather data in a folder and convert them to csv
def excel2csv():
    print 'read excel file'
    # filename = os.getcwd() + '/input/FY/WeatherGSA.xlsx'
    filelist = glob.glob(os.getcwd() + '/input/FY/weather/*.xlsx')
    print filelist
    for f in filelist:
        filename = f[f.rfind('/') + 1: f.rfind('.')]
        df = pd.read_excel(f, sheetname=0)
        df.to_csv(homedir + 'weatherinput/{0}.csv'.format(filename))
        print 'output' + (homedir +
                          'weatherinput/{0}.csv'.format(filename))

# check and remove nan values for weather data
def check_data():
    filelist = glob.glob(homedir + 'weatherinput/*.csv')
    print filelist

    def is_string(s):
        # float, with possible trailing e-0x
        pattern = re.compile('-?[0-9]{0,3}\.?[0-9]{0,20}(e-[0-9]{2})?$')
        if type(s) is str:
            return (not re.match(pattern, s))
        else:
            return False

    for f in filelist:
        df = pd.read_csv(f)
        # weather data is of the format:
        #                       KBOS
        #                       \\128.2.109.159\WeatherUnderground/KBOS/Temperature
        #
        #   2012-Sep-01 0:00:00 80.0999984741211
        #   2012-Sep-01 1:00:00 79.9899978637695
        #   2012-Sep-01 2:00:00 79

        # drop rows with empty time stamp
        df.dropna(subset=['Unnamed: 0'], inplace=True)
        cols = list(df)[1:]
        print 'number of columns before dropna: {0}'.format(len(cols))
        err_strings = []
        for col in cols:
            col_value = df[col].tolist()
            str_value = [x for x in col_value if is_string(x)]
            str_set = set(str_value)
            if len(str_set) != 0:
                err_strings += list(str_set)
        err_str_set = set(err_strings)
        print 'The set of error string: {0}'.format(set(err_str_set))

        df.replace(list(err_str_set), np.nan, inplace=True)
        print len(df)
        df.drop(len(df), axis=0, inplace=True)
        print len(df)
        df.dropna(axis=1, how='any', inplace=True)
        clean_cols = list(df)
        print 'number of columns after dropna: {0}'.format(len(clean_cols))
        assert('KDMH' not in df)
        df.to_csv(f.replace('.csv', '_nonan.csv'), index=False)
    return
    
# requires input files have the same time duration
def union_weatherinput():
    filelist = glob.glob(homedir + 'weatherinput/*_nonan.csv')
    print filelist
    df_base = pd.read_csv(filelist[0])
    existing_cols = list(df_base)
    print 'original number of stations: {0}'.format(len(existing_cols))
    length = len(df_base)
    other = filelist[1:]
    for f in filelist:
        df_i = pd.read_csv(f)
        print f[f.rfind('/') + 1:]
        assert(len(df_i) == length)
        col_to_use = list(df_i.columns - df_base.columns)
        col_to_use.append('Unnamed: 0')
        df_base = pd.merge(df_base, df_i[col_to_use], on='Unnamed: 0',
                           how='inner')
        print(len(df_base))
        existing_cols = list(df_base)
        print 'number of stations after merge: {0}'.format(len(existing_cols))
    cols = list(df_base)
    assert(len(cols) == len(set(cols)))
    df_base.to_csv(homedir + 'weatherData_nonan.csv', index=False)

def get_mean_temp():
    outfile = (homedir + 'weatherData_meanTemp.csv')
    print 'write mean temperature to {0}'.format(outfile)
    df = pd.read_csv(homedir + 'weatherData_nonan.csv')
    df.set_index(pd.DatetimeIndex(df['Unnamed: 0']), inplace=True)
    df.resample('M', how = 'mean').to_csv(outfile, index=True)

def get_DD_itg(base, theme):
    df = pd.read_csv(homedir + 'weatherData_nonan.csv')
    df.set_index(pd.DatetimeIndex(df['Unnamed: 0']), inplace=True)
    df_hour = df.resample('H', how = 'mean')

    for col in df_hour:
        if theme == 'HDD':
            df_hour[col] = df_hour[col].map(lambda x: 0 if x >= base else base - x)
        else:
            df_hour[col] = df_hour[col].map(lambda x: 0 if x <= base else x - base)
    df_day = df_hour.resample('D', how = 'mean')
    print 'base temperature: {0}'.format(base)
    print df_day['KBOS'].head()
    df_day.to_csv(homedir + 'degreeday/Day{1}_itg_{0}F.csv'.format(int(base), theme))
    df_month = df_day.resample('M', how = 'sum').to_csv(homedir + 'degreeday/{1}_itg_{0}F.csv'.format(int(base), theme))

# read energy of building b
def read_energy(b):
    filelist = glob.glob(os.getcwd() + '/csv_FY/single_eui_cal/{0}*.csv'.format(b))
    dfs = [pd.read_csv(csv) for csv in filelist]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.sort(columns=['year', 'month'], inplace=True)
    return df_all

# read temperature record from Oct. 2012 to Sep. 2015
def read_temperature():
    df = pd.read_csv(homedir + 'weatherData_meanTemp.csv')
    df.drop(0, axis=0, inplace=True)
    return df

# read icao cade (4-alphabetical char) of weather station
def read_icao():
    names = ['Block Number', 'Station Number', 'ICAO Location', 'Indicator',
             'Place Name', 'State', 'Country Name', 'WMO Region',
             'Station Latitude', 'Station Longitude', 'Upper Air Latitude',
             'Upper Air Longitude', 'Station Elevation (Ha)',
             'Upper Air Elevation (Hp)', 'RBSN indicator']
    df = pd.read_csv(homedir + 'nsd_bbsss.txt', sep=';',
                     header=None, names=names)

    df['WMO ID'] = df.apply(lambda row: str(row['Block Number']).zfill(2) + str(row['Station Number']).zfill(3), axis=1)
    df = df[['WMO ID', 'ICAO Location']]
    return df

def read_ghcnd():
    names = ['ID', 'LATITUDE', 'LONGITUD', 'ELEVATION', 'STATE', 'NAME',
             'GSN FLAG', 'HCN/CRN FLAG', 'WMO ID']
    filename = homedir + 'ghcnd-stations.txt'
    outfile = homedir + 'ghcnd-stations-delim.txt'
    with open (filename, 'r') as rd:
        lines = rd.readlines()
    with open (outfile ,'w+') as wt:
        for line in lines:
            line_list = list(line)
            for i in [11, 20, 30, 37, 40, 71, 75, 79]:
                line_list[i] = ','
            new_line = ''.join(line_list)
            wt.write(new_line)
    df = pd.read_csv(homedir + 'ghcnd-stations-delim.txt',
                     header=None, names=names)
    df = df[['ID', 'WMO ID']]
    return df

# read climate normal
def read_ncdc():
    import calendar
    names = ['ID'] + [calendar.month_abbr[i] for i in range(1, 13)]
    df = pd.read_csv(homedir + 'mly-tavg-normal.txt',
                     delim_whitespace=True, header=None, names=names)
    # checked, no special value in the file
    for col in df:
        if col != 'ID':
            df[col] = df[col].map(lambda x: int(x[:-1])/10.0)
    return df

# return lookup table of ical to ghcnd
def read_temp_norm():
    df_icao = read_icao()
    df_ghcnd = read_ghcnd()
    #df_ghcnd.to_csv(homedir + 'ghcnd.csv', index=False)
    df_merge = pd.merge(df_icao, df_ghcnd, on='WMO ID', how='left')
    df_merge = df_merge[df_merge['ICAO Location'] != '----']
    #df_merge.to_csv(homedir + 'icao_ghcnd.csv',
    #                index=False)
    df_temp = read_ncdc()
    df_all = pd.merge(df_merge, df_temp, on='ID', how='left')
    #df_all.to_csv(homedir + 'icao_ghcnd_ncdc.csv',
    #              index=False)
    return df_all

def plot_energy_temp(df_energy, df_temp, theme, b, s):
    df = pd.DataFrame({'energy': df_energy[theme], 'temp': df_temp[s]})
    sns.regplot('temp', 'energy', data=df, fit_reg=False)
    P.savefig(os.getcwd() + '/plot_FY_weather/{2}/{0}_{1}.png'.format(b, s, theme), dpi = 150)
    plt.title('Temperature-{0} plot: {1}, {2}'.format(theme, b, s))
    plt.close()
    return

def plot_energy_temp_byyear_2015(theme):
    sns.set_palette(sns.color_palette('Set2', 27))
    sns.mpl.rc("figure", figsize=(10,5))
    cat_df = pd.read_csv(os.getcwd() + '/csv_FY/join_cal/join_2015.csv')
    cat_dict = dict(zip(cat_df['Building Number'].tolist(),
                        cat_df['Cat'].tolist()))
    filelist = glob.glob(os.getcwd() + '/csv_FY/energy_temperature_select/*_{0}.csv'.format(title_dict[theme]))
    def getname(dirname):
        id1 = dirname.find('select') + len('select') + 1
        return dirname[id1: id1 + 8]
    buildings = [getname(f) for f in filelist]
    dfs = [pd.read_csv(csv) for csv in filelist]
    dfs = [df[df['Fiscal Year'] == 2015] for df in dfs]
    euis = [round(df[theme].sum(), 2) for df in dfs]
    sorted_bedf = sorted(zip(buildings, euis, dfs), key=lambda x: x[1], reverse=True)
    buildings = [x[0] for x in sorted_bedf]
    euis = [x[1] for x in sorted_bedf]
    dfs = [x[2] for x in sorted_bedf]
    lines = []
    for i in range(len(buildings)):
        df = dfs[i]
        df.sort(['temperature', theme], inplace=True)
        line, = plt.plot(df['temperature'], df[theme])
        lines.append(line)
    labels = ['{0}: {1} kBtu/sq.ft*year_{2}'.format(b, e, cat_dict[b]) for (b, e) in zip(buildings, euis)]
    plt.title('Temperature-{0} plot: 27 Building, Fiscal Year 2015'.format(title_dict[theme]))
    plt.xlabel(xlabel_temp, fontsize=12)
    plt.ylabel(ylabel_dict[theme], fontsize=12)
    plt.legend(lines, labels, bbox_to_anchor=(0.2, 1), prop={'size':6})
    P.savefig(os.getcwd() + '/plot_FY_weather/27building_{0}_2015_trunc.png'.format(theme), dpi = 150)
    #P.savefig(os.getcwd() + '/plot_FY_weather/27building_{0}_2015.png'.format(theme), dpi = 150)
    plt.close()
    return
    
#ld: line or dot plot
def plot_energy_temp_byyear(df_energy, df_temp, df_hdd, df_cdd, theme,
                            b, s, ld, kind, remove0):
    sns.set_palette(sns.color_palette('Set2', 9))
    sns.mpl.rc("figure", figsize=(10,5))
    df = df_energy
    df['temp'] = df_temp[s].tolist()
    df['hdd'] = df_hdd[s].tolist()
    df['hdd'] = df['hdd'] * (-1.0)
    df['cdd'] = df_cdd[s].tolist()
    df.to_csv(os.getcwd() + '/csv_FY/energy_temperature_select/{0}_{1}_{2}.csv'.format(b, s, title_dict[theme]), index=False)
    df1 = df.copy()
    df1['dd'] = df1['hdd']
    df2 = df.copy()
    df2['dd'] = df2['cdd']
    df3 = pd.concat([df1, df2], ignore_index=True)
    if kind != 'all':
        print df[kind].head()
        df = df[df[kind] != 0.0]
        if ld == 'line':
            gr = df.groupby('Fiscal Year')
            lines = []
            for name, group in gr:
                print (name, kind)
                group.sort([kind, theme], inplace=True)
                group = group[[kind, theme]]
                line, = plt.plot(group[kind], group[theme])
                lines.append(line)
                #print 'Building: {0}, year: {1}, {2} {3} [kbtu/sq.ft.]'.format(b, int(name), round(group[theme].sum(), 2), title_dict[theme])
        else:
            if kind == 'cdd':
                sns.set_palette(sns.color_palette('Blues'))
            elif kind == 'hdd':
                sns.set_palette(sns.color_palette('Oranges'))
            sns.lmplot(x=kind, y=theme, hue='Fiscal Year', data=df, fit_reg=True)
            x = np.array(df[kind])
            y = np.array(df[theme])
            t_min = df[kind].min()
            t_max = df[kind].max()
            xd = np.r_[t_min:t_max:1]
            k1 = smooth.NonParamRegression(x, y, method=npr_methods.LocalPolynomialKernel(q=1))
            plt.plot(xd, k1(xd), '-', color=sns.color_palette('Set2')[5])
            plt.xlabel(xlabel_dict[kind], fontsize=12)
            plt.ylabel(ylabel_dict[theme], fontsize=12)
            plt.title('{3}-{0} plot: Building {1}, Station {2}'.format(title_dict[theme], b, s, kind_dict[kind]))
    else:
        if ld == 'line':
            gr = df.groupby('Fiscal Year')
            lines = []
            for name, group in gr:
                print (name, kind)
                group_elec = group.sort(['cdd', 'eui_elec'])
                group_gas = group.sort(['hdd', 'eui_gas'])
                # offset temperature to 0F
                group['temp'] = group['temp'] - 65.0
                group['temp_dd'] = group.apply(lambda r: r['hdd'] if r['temp'] < 0 else r['cdd'], axis=1)
                group_temp = group.sort(['temp_dd', 'eui'])
                if remove0:
                    group_elec = group_elec[group_elec['cdd'] >= 10]
                    group_gas = group_gas[group_gas['hdd'] <= -10]
                group_temp = group.sort(['temp', 'eui'])
                line_elec, = plt.plot(group_elec['cdd'],
                                      group_elec['eui_elec'])
                line_gas, = plt.plot(group_gas['hdd'],
                                     group_gas['eui_gas'])
                line_temp, = plt.plot(group_temp['temp_dd'], group_temp['eui'])
                lines.append(line_elec)
                lines.append(line_gas)
                lines.append(line_temp)
            plt.ylabel(ylabel_dict[theme], fontsize=12)
            plt.title('{3}-{0} plot: Building {1}, Station {2}'.format(title_dict[theme], b, s, kind_dict[kind]))
        else:
            base_load = df3.copy()
            gr_base = base_load.groupby('Fiscal Year')
            base_gas_dict = {}
            base_elec_dict = {}
            for name, group in gr_base:
                tempdf_gas = group.copy()
                tempdf_elec = group.copy()
                tempdf_gas['base_month'] = tempdf_gas['month'].map(lambda x: True if x == 12 or x < 3 else False)
                tempdf_gas = tempdf_gas[tempdf_gas['base_month'] == True]
                print tempdf_gas
                base_gas_dict[name] = tempdf_gas['eui_gas'].mean()

                tempdf_elec['base_month'] = \
                  ap(lambda x: True \
                        if (6 <= x and x <= 8) else False)
                tempdf_elec = tempdf_elec[tempdf_elec['base_month'] == True]
                print tempdf_elec
                base_elec_dict[name] = tempdf_elec['eui_elec'].mean()

            print base_elec_dict
            print base_gas_dict
            if remove0:
                df3 = df3[df3['dd'].abs() >= 30]
            df_elec = df3.copy()
            df_elec['kind'] = 'Electricity'
            df_elec['eui_plot'] = df_elec['eui_elec']
            df_gas = df3.copy()
            df_gas['kind'] = 'Natural Gas'
            df_gas['eui_plot'] = df_gas.apply(lambda r: r['eui_gas'] + base_elec_dict[r['Fiscal Year']], axis=1)
            df_total = df3.copy()
            df_total['kind'] = 'Total'
            df_total['eui_plot'] = df_gas['eui']
            df_base_elec = df3.copy()
            df_base_elec['kind'] = 'Base-Electricity'
            df_base_elec['eui_plot'] = df_base_elec['Fiscal Year'].map(lambda x: base_elec_dict[x])
            df_base_gas = df3.copy()
            df_base_gas['kind'] = 'Base-Gas'
            df_base_gas['eui_plot'] = df_base_gas['Fiscal Year'].apply(lambda r: base_gas_dict[r] + base_elec_dict[r])
            df_all = pd.concat([df_elec, df_gas, df_total, df_base_gas, df_base_elec], ignore_index=True)
            #df_all = pd.concat([df_elec, df_gas, df_total], ignore_index=True)
            g = sns.lmplot(x='dd', y='eui_plot', data=df_all, col='Fiscal Year', hue = 'kind', fit_reg=True, truncate=True, lowess=True)
            #plt.xlabel(xlabel_dict[kind], fontsize=12)
            g = g.set_axis_labels(xlabel_dict[kind],
                                  ylabel_dict[kind])
    plt.ylim((0, 7))
    if ld == 'line':
        if kind != 'all':
            P.savefig(os.getcwd() + '/plot_FY_weather/eui_{3}/{2}_byyear/{0}_{1}.png'.format(b, s, theme, kind), dpi = 150)
        else:
            years = ['2013', '2014', '2015']
            line_labels = ['Electricity', 'Gas', 'Total']
            labels = reduce(lambda x, y: x + y, [['{0}-{1}'.format(x, y) for y in line_labels] for x in years])
            plt.legend(lines, labels)
            P.savefig(os.getcwd() + '/plot_FY_weather/eui_{2}/{2}_byyear/{0}_{1}.png'.format(b, s, kind), dpi = 150)

    else:
        if kind != 'all':
            print (b, s, theme, kind)
            P.savefig(os.getcwd() + '/plot_FY_weather/eui_{3}/{2}_byyear_dot/{0}_{1}.png'.format(b, s, theme, kind), dpi = 75)
        else:
            P.savefig(os.getcwd() + '/plot_FY_weather/eui_{2}/{2}_byyear_dot/{0}_{1}.png'.format(b, s, kind), dpi = 75)
    plt.close()

def regression_hdd(t, s, df, theme):
    df_temp = pd.read_csv(homedir + 'weatherData_HDD_{0}F.csv'.format(t))
    df_temp.drop(0, inplace=True)
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_temp[s], df[theme])
    return (slope, intercept, r_value * r_value, t)

def regression_gas_temp(b, s, df, theme):
    x = np.array(df['temperature'])
    y = np.array(df[theme])
    def piecewise_linear(x, x0, y0, k1, k2):
        return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
    p , e = optimize.curve_fit(piecewise_linear, x, y)
    t_min = df['temperature'].min()
    t_max = df['temperature'].max()
    xd = np.linspace(t_min, t_max, 15)
    plt.plot(x, y, "o")
    plt.plot(xd, piecewise_linear(xd, *p))
    P.savefig(os.getcwd() + '/plot_FY_weather/eui_gas_piece/{0}_{1}.png'.format(b, s), dpi=150)
    plt.close()
    return

def regression_gas_kernel(b, s, df, theme):
    x = np.array(df['temperature'])
    y = np.array(df[theme])
    t_min = df['temperature'].min()
    t_max = df['temperature'].max()
    xd = np.r_[t_min:t_max:1]
    k1 = smooth.NonParamRegression(x, y, method=npr_methods.LocalPolynomialKernel(q=1))
    plt.plot(x, y, "o")
    plt.plot(xd, k1(xd))
    plt.xlabel(xlabel_temp, fontsize=12)
    plt.ylabel(ylabel_dict[theme], fontsize=12)
    plt.title('Kernel Regression Fit {0} - Temperature Plot\n Building {1}, Station {2}'.format(title_dict[theme], b, s), fontsize=15)
    P.savefig(os.getcwd() + '/plot_FY_weather/eui_gas_kernel/{0}_{1}.png'.format(b, s), dpi=150)
    plt.close()
    return k1

# Smoothing spline
def plot_normal(df, theme, b, s):
    sns.set_palette(sns.color_palette('Paired', 8))
    gr = df.groupby('year')
    lines = []
    for name, group in gr:
        ori,  = plt.plot(group['month'], group[theme])
        norm, = plt.plot(group['month'], group['e_norm'])
        lines.append(ori)
        lines.append(norm)
    plt.legend(lines, ['2012_ori', '2012_norm', '2013_ori', '2013_norm', '2014_ori', '2014_norm', '2015_ori', '2015_norm'], bbox_to_anchor=(0.2, 1))
    plt.title('{0}\nBuilding {1}'.format(title_weather[theme], b), fontsize=15, x = 0.5, y = 1)
    plt.xlim((1, 12))
    plt.xlabel('Month', fontsize=12)
    plt.xticks(range(1, 13), [calendar.month_abbr[m] for m in range(1, 13)])
    plt.ylabel(ylabel_dict[theme], fontsize=12)
    P.savefig(os.getcwd() + '/plot_FY_weather/{2}_ori_norm/{0}_{1}.png'.format(b, s, theme), dpi=150)
    plt.close()

def plot_normal_only(df, theme, b, s):
    sns.set_palette(sns.color_palette('Set2', 4))
    gr = df.groupby('year')
    lines = []
    for name, group in gr:
        norm, = plt.plot(group['month'], group['e_norm'])
        lines.append(norm)
    plt.legend(lines, ['2012_norm', '2013_norm', '2014_norm', '2015_norm'], bbox_to_anchor=(0.2, 1))
    plt.title('{0}\nBuilding {1}'.format(title_dict_3[theme], b), fontsize=15, x = 0.5, y = 1)
    plt.xlim((1, 12))
    plt.xlabel('Month', fontsize=12)
    plt.xticks(range(1, 13), [calendar.month_abbr[m] for m in range(1, 13)])
    plt.ylabel(ylabel_dict[theme], fontsize=12)
    P.savefig(os.getcwd() + '/plot_FY_weather/{2}_norm/{0}_{1}.png'.format(b, s, theme), dpi=150)
    plt.close()

def regression_hdd_changebase(df_all, theme):
    fit_result = []
    for t in range(55, 75):
        df_temp = df_all.copy()
        df_temp = df_temp['temperature'] < t
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_temp[t], df_temp[theme])
        fit_result.append((slope, intercept, r_value, p_value, t))
    best_fit = sorted(fit_result, key = lambda x: x[2])[0]
    return best_fit

def calculate_piecewise(theme):
    b_list = ['FL0067ZZ', 'KY0045ZZ', 'MA0131ZZ', 'CO0039ZZ']
    bs_pair = read_building_weather()
    bs_pair = [x for x in bs_pair if x[0] in b_list]
    # Oct 2012 to Sep 2013
    df_temperature = read_temperature()
    for b, s in bs_pair:
        print (b, s)
        df_energy = read_energy(b)[['Fiscal Year', 'year', 'month', 'eui_elec', 'eui_gas', 'eui']]
        # read temperature and degree day
        df_temp_dday = read_temp_dday(s)
        df_all = pd.merge(df_energy, df_temp_dday, on='month', how = 'inner')
        reg_result = regression_hdd_changebase(df_energy, df_temp_dday, theme)
        t_change = reg[-1]
        r_value = reg[2]
        df_base = df_all.copy()
        df_base = df_base[df_base['temperature'] > t_change]
        baseload = df_base[theme].mean()
        df_slope = df_all.copy()
        df_slope = df_slope[df_slope['temperature'] < t_change]
        sns.regplot(x = '{0}'.format(t_change), y = theme, data = df_slope)
        plt.show()
        plt.close()
        print baseload
    return

def write_temp_dd():
    bs_pair = read_building_weather()
    stations = [x[1] for x in bs_pair]
    print len(stations)
    df_temp = read_temperature()
    hdd_filelist = glob.glob(homedir + 'weatherData_HDD_itg_*')
    dfs_hdd = [pd.read_csv(f) for f in hdd_filelist]
    base_temp = [f[f.rfind('_') + 1: f.rfind('.')] for f in hdd_filelist]
    length = len(dfs_hdd)
    print length
    for i in range(length):
        s = stations[i]
        df_s = df_temp[[s]].copy()
        for base in base_temp:
            df_s[base] = dfs_hdd[i][s]
        df_s.to_csv(homedir + 'temp_dd/{0}.csv'.format(s))

def get_gsalink_set():
    df = pd.read_csv(os.getcwd() + '/input/FY/GSAlink 81 Buildings Updated 9_22_15.csv')
    return list(set(df['Building ID'].tolist()))

def calculate(theme, method):
    bs_pair = read_building_weather('building_station_lookup.csv',
                                    'Building Number', 'Weather Station')
    study_set = get_gsalink_set()
    bs_pair = [x for x in bs_pair if x[0] in study_set]
    df_temperature = read_temperature()
    df_temp_norm = read_temp_norm()
    df_temp_norm.drop(['WMO ID', 'ID'], axis=1, inplace=True)
    df_hdd_65 = pd.read_csv(homedir + 'weatherData_HDD_itg_65F.csv')
    df_hdd_65.drop(0, axis=0, inplace=True)
    df_cdh_65 = pd.read_csv(homedir + 'weatherData_CDD_itg_65F.csv')
    df_cdh_65.drop(0, axis=0, inplace=True)

    '''
    t_norm = df_temp_norm[df_temp_norm['ICAO Location'] == s]
    print len(t_norm)
    if len(t_norm) == 0:
        print s
        continue
    t_norm_list = list(list(t_norm.itertuples())[0])[2:]
    '''
    for b, s in bs_pair[:1]:
        print (b, s)
        df_energy = read_energy(b)[['Fiscal Year', 'year', 'month', 'eui_elec', 'eui_gas', 'eui']]
        # 2012 to 2015 data
        df_energy = df_energy[-48:] # remove this when weather data available
        df_t = df_temperature[[s]][-48:]
        df_h = df_hdd_65[[s]][-48:]
        df_c = df_cdh_65[[s]][-48:]
        plot_energy_temp_byyear(df_energy, df_t, df_h, df_c, theme,
                                b, s, 'dot', 'all', True)
        '''
        if theme == 'eui_gas':
            plot_energy_temp_byyear(df_energy, df_t, df_h, df_c, theme, b, s,
                    'dot', 'hdd', True)
        elif theme == 'eui_elec':
            plot_energy_temp_byyear(df_energy, df_t, df_h, df_c, theme, b, s,
                    'dot', 'cdd', True)
        '''
        '''
        plot_energy_temp_byyear(df_energy, df_t, df_h, df_c, theme,
                                b, s, 'line', 'all', True)
        df = df_energy
        df['temperature'] = df_temperature[s].tolist()
        if theme == 'eui_gas':
            if method == 'hdd':
                reg_list = [regression_hdd(t, s, df, theme) for t in [40, 45, 50, 55, 57, 60, 65]]
                reg_list = sorted(reg_list, key = lambda x: x[2], reverse=True)
                print reg_list
                (slope, intercept, r_sqr, t) = reg_list[0]
                print (round(r_sqr, 2), t)
            elif method == 'temp':
                t_min = df['temperature'].min()
                t_max = df['temperature'].max()
                dx = (t_max - t_min) / 10
                reg = regression_gas_temp(b, s, df, theme)
            elif method == 'kernel':
                reg = regression_gas_kernel(b, s, df, theme)
                df['t_norm'] = np.array((t_norm_list * 4)[9: 9+36])
                df['e_norm'] = df.apply(lambda r: r[theme]/r['temperature']*r['t_norm'], axis=1)
                plot_normal(df, theme, b, s)
                plot_normal_only(df, theme, b, s)
        else:
            reg = regression_gas_kernel(b, s, df, theme)
            # starting from october
            df['t_norm'] = np.array((t_norm_list * 4)[9: 9+36])
            df['e_norm'] = df.apply(lambda r: r[theme]/r['temperature']*r['t_norm'], axis=1)
            plot_normal(df, theme, b, s)
            plot_normal_only(df, theme, b, s)
        '''
from geopy.geocoders import Nominatim
from geopy.distance import vincenty

def getICAO(StateAbbr, City, Address, zipcode, df_lookup):
    df = df_lookup[df_lookup['StateAbbr'] == StateAbbr]
    City = City.upper()
    cities = df['City'].tolist()
    counter = 0
    if City in cities:
        df.set_index('City', inplace=True)
        return df.ix[City, 'ICAO']
    else:
        geolocator = Nominatim()
        location = geolocator.geocode('{0},{1},{2},{3}'.format(Address, City, StateAbbr, zipcode))
        #location = geolocator.geocode('{0}'.format(zipcode))
        if location == None:
            return 'Not Found'
        print location
        df['distance'] = df.apply(lambda r: vincenty((location.latitude, location.longitude), (r['Lat'], r['Long'])).miles, axis=1)
        min_distance = df['distance'].min()
        df = df[df['distance'] == min_distance]
        return df['ICAO'].iloc[0]

def pm_static_info_2_station():
    df = pd.read_csv(os.getcwd() + '/csv/all_column/sheet-0-all_col.csv')
    df = df[['Property Name', 'Street Address', 'City/Municipality',
             'State/Province', 'Postal Code']]
    df['Property Name'] = df['Property Name'].map(lambda x: x.partition(' ')[0][:8])
    df['Postal Code'] = df['Postal Code'].map(lambda x: x[:5])

    df_state = pd.read_csv(os.getcwd() + '/input/FY/state2abbr.csv')
    df_state = df_state[['State', 'Postal']]
    df_all = pd.merge(df, df_state, left_on='State/Province', right_on='State', how='left')
    df_all.drop('State', axis=1, inplace=True)
    df_all.rename(columns={'Property Name' : 'Building Number',
                           'City/Municipality' : 'City',
                           'State/Province' : 'State',
                           'Postal': 'StateAbbr',
                           'Postal Code': 'zipcode'}, inplace=True)
    df_stationlookup = pd.read_csv(homedir + 'weatherinput/Weather Station Mapping.csv')
    df_stationlookup = df_stationlookup[['City', 'ICAO', 'StateAbbr', 'Lat', 'Long']]
    step = 50
    dfs = [df_all[i * step: (i + 1) * step] for i in range(0, len(df_all)/step)]
    for i in range(len(dfs))[:1]:
        print i, len(dfs[i])
        starttime = time.time()
        dfs[i]['Weather Station'] = dfs[i].apply(lambda r: getICAO(r['StateAbbr'], r['City'], r['Street Address'], r['zipcode'], df_stationlookup), axis=1)
        endtime = time.time()
        # FIXME: print non- "Not Found" result count
        dfs[i].to_csv(homedir + 'location_info_{0}.csv'.format(i), index=False)
        print endtime-starttime
    print 'end'

# generate building station lookup table from state and city
def building_to_station_fromlocation():
    df_location = pm_static_info_2_station()
    '''
    df = pd.read_csv(os.getcwd() + '/csv_FY/filter_bit/fis/indicator_all.csv')
    df = df[df['good_area'] == 1]
    df = df[['Building Number']]
    df_all = pd.merge(df, df_location, on='Building Number', how='left')
    print df_all.head()
    df_all.to_csv(homedir + 'building_station_lookup_loc.csv', index=False)
    '''
    return

def building_to_station():
    df = pd.read_csv(os.getcwd() + '/csv_FY/filter_bit/fis/indicator_all.csv')
    df = df[df['good_area'] == 1]
    good_building = set(df['Building Number'].tolist())
    good_station = set(list(pd.read_csv(homedir + 'weatherData_meanTemp.csv')))
    station_info = pd.read_csv(homedir + 'weatherStation.csv')
    station_info = station_info[['Building Number', 'Weather Station']]
    print len(station_info)
    station_info.dropna(inplace=True)
    print len(station_info)
    station_info['Weather Station'] = station_info['Weather Station'].map(lambda x: x.replace(' ', ''))
    station_info.drop_duplicates(cols='Building Number', inplace=True)
    #print '#{0}#'.format(station_info.iloc[60, 1])
    station_info = station_info[station_info['Weather Station'].isin(good_station)]
    station_info = station_info[station_info['Building Number'].isin(good_building)]
    print 'number of building with station {0}'.format(len(station_info))
    station_info.to_csv(homedir + 'building_station_lookup.csv', index=False)

# read building_station lookup table to pair list
def read_building_weather(filename, id_col, weather_col):
    df = pd.read_csv(homedir + filename)
    return zip(df[id_col].tolist(), df[weather_col].tolist())

def sep_dd(kind, low, high):
    print 'start separating degree day'
    files_dd = [homedir + 'degreeday/{1}_itg_{0}F.csv'.format(i, kind)\
                for i in range(low, high)]
    keys = ['{0}F'.format(x) for x in range(low, high)]
    dfs_dd = [pd.read_csv(f) for f in files_dd]
    # time column
    col_time = dfs_dd[0].iloc[:, 0]
    stations = list(dfs_dd[0])[1:]
    for s in stations:
        dd_list = [df[s].tolist() for df in dfs_dd]
        d = dict(zip(keys, dd_list))
        d['timestamp'] = col_time
        df_s = pd.DataFrame(d)
        df_s['year'] = df_s['timestamp'].map(lambda x: x[:4])
        df_s['month'] = df_s['timestamp'].map(lambda x: x[5: 7])
        re_order_col = list(df_s)
        re_order_col.remove('year')
        re_order_col.remove('month')
        re_order_col.remove('timestamp')
        re_order_col = ['timestamp', 'year', 'month'] + re_order_col
        df_s = df_s[re_order_col]
        df_s.to_csv(homedir + 'station_dd/{0}_{1}.csv'.format(s, kind),
                    index=False)
    print 'end separating degree day'

def sep_temp():
    print 'start separating temperature'
    df_temp = pd.read_csv(homedir + 'weatherData_meanTemp.csv')
    col_time = df_temp.iloc[:, 0]
    stations = list(df_temp)[1:]
    for s in stations:
        d = {s: df_temp[s], 'timestamp': col_time}
        df_s = pd.DataFrame(d)
        df_s['year'] = df_s['timestamp'].map(lambda x: x[:4])
        df_s['month'] = df_s['timestamp'].map(lambda x: int(x[5: 7]))
        re_order_col = list(df_s)
        re_order_col.remove(s)
        re_order_col.append(s)
        df_s = df_s[re_order_col]
        df_s.to_csv(homedir + 'station_temp/{0}.csv'.format(s),
                    index=False)
    print 'end separating temperature'

def read_mean_temp(s):
    df = pd.read_csv(homedir + 'station_temp/{0}.csv'.format(s))
    return df

def join_building_temp():
    bs_pair = read_building_weather('building_station_lookup.csv',
                                    'Building Number', 'Weather Station')
    for (b, s) in bs_pair:
        print (b, s)
        df_energy = read_energy(b)
        df_temp = read_mean_temp(s)
        df = pd.merge(df_energy, df_temp, how='inner', on=['year', 'month'])
        df.to_csv(homedir + 'energy_temp/{0}_{1}.csv'.format(b, s),
                  index=False)

# BOOKMARK
def plot_building_temp(b, s):
    print 'not implemented'
    return

def process_weatherfile():
    # slow, comment out once you have the output
    # excel2csv()

    # check_data()
    # union_weatherinput()
    # get_mean_temp()
    # calculate_dd()
    # sep_dd('HDD', 40, 81)
    # sep_dd('CDD', 40, 81)
    # sep_temp()

    # building_to_station()
    building_to_station_fromlocation() # need Shilpi's file
    # join_building_temp()
    # plot_building_temp()
    return

def calculate_dd():
    for base in range(40, 81):
        get_DD_itg(base, 'HDD')
        get_DD_itg(base, 'CDD')

def join_dd_temp_energy(b, s, kind):
    df_eng_temp = pd.read_csv(homedir +
                              'energy_temp/{0}_{1}.csv'.format(b, s))
    df_dd = pd.read_csv(homedir + 
                        'station_dd/{0}_{1}.csv'.format(s, kind))
    df_all = pd.merge(df_eng_temp, df_dd, on=['year', 'month'],
                      how='inner')
    df_all.to_csv(homedir + '/dd_temp_eng/{2}_{0}_{1}.csv'.format(b, s, kind), index=False)

# kind: CDD, HDD
def opt_lireg(b, s, kind):
    df_eng_temp = pd.read_csv(homedir +
                              'energy_temp/{0}_{1}.csv'.format(b, s))
    df_dd = pd.read_csv(homedir + 
                        'station_dd/{0}_{1}.csv'.format(s, kind))
    df_all = pd.merge(df_eng_temp, df_dd, on=['year', 'month'],
                      how='inner')
    df_all = df_all[df_all['year'] < 2013]
    dd_list = ['{0}F'.format(x) for x in range(40, 81)]
    if kind == 'CDD':
        theme = 'eui_elec'
    else:
        theme = 'eui_gas'

    results = []
    for col in dd_list:
        lean_x = df_all[col]
        lean_y = df_all[theme]
        slope, intercept, r_value, p_value, std_err = \
            stats.linregress(lean_x, lean_y)
        results.append([slope, intercept, r_value, col])
    ordered_result = sorted(results, key=lambda x: x[2], reverse=True)
    print ordered_result[0]
    '''
    base_temp = ordered_result[0][3]
    base_load = ordered_result[0][1]
    plot_temp_fit(df_all, base_temp, b, s, kind, theme, base_load)
    '''
    slope_opt, intercept_opt, r_opt, col_opt = ordered_result[0]
    plot_dd_fit(df_all, slope_opt, intercept_opt, r_opt, col_opt,
                theme, kind, b, s)
    return ordered_result[0]

def plot_dd_fit(df_all, slope, intercept, r, xF, theme, kind, b, s):
    x = df_all[xF]
    y = df_all[theme]
    xd = [0, x.max()]
    yd = [intercept, slope * x.max() + intercept]
    sns.set_style("white")
    sns.set_palette("Set2")
    sns.set_context("talk", font_scale=1.5)
    bx = plt.axes()
    bx.annotate('y = {0} x + {1}\nR^2: {2}'.format(round(slope, 3), 
                                                   round(intercept, 3),
                                                   round(r * r, 3)),
                xy = (x.max() * 0.1, y.max() * 0.9),
                xytext = (x.max() * 0.05, y.max() * 0.95), fontsize=20)
    bx.plot(x, y, 'o', xd, yd, '-')
    plt.ylim((0, y.max() * 1.1))
    plt.title('{0} - {1} Plot'.format(title_dict[theme], kind))
    plt.suptitle('Building {0}, Station {1}'.format(b, s))
    plt.xlabel('{0} Deg F'.format(kind))
    plt.ylabel(ylabel_dict[theme])
    P.savefig(os.getcwd() + '/plot_FY_weather/dd_energy/{2}/{0}_{1}.png'.format(b, s, theme), dpi = 150)
    plt.close()

# s: station id
def plot_temp_fit(df_all, basetemp, b, s, kind, theme, base_load):
    print (basetemp, b, s, kind, theme)
    x = df_all[s]
    y = df_all[theme]
    tmin = df_all[s].min()
    tmax = df_all[s].max()
    pairs = zip(x, y)
    base = int(basetemp[:2])
    left = [p for p in pairs if p[0] < base]
    right = [p for p in pairs if p[0] >= base]
    left_x = [p[0] for p in left]
    left_y = [p[1] for p in left]
    right_x = [p[0] for p in right]
    right_y = [p[1] for p in right]
    if len(left) > 0:
        left_ave = sum(left_y)/len(left)
        slope_l, intercept_l, r_value_l, p_value_l, std_err_l = \
            stats.linregress(left_x, left_y)
    if len(right) > 0:
        right_ave = sum(right_y)/len(right)
        slope_r, intercept_r, r_value_r, p_value_r, std_err_r = \
            stats.linregress(right_x, right_y)
    def fit(x, slope, intercept):
        return np.array([slope * xi + intercept for xi in x])
    def ave(x, length):
        average = sum(x) / len(x)
        return np.array([average] * length)
    sns.set_style("white")
    sns.set_palette("Set2")
    sns.set_context("talk", font_scale=1.5)
    #sns.mpl.rc("figure", figsize=(10,5.5))
    plot_x_left = np.array(left_x)
    plot_y_left = np.array(left_y)
    plot_tmin_left = tmin
    plot_tmax_left = base
    xd_left = np.r_[plot_tmin_left:plot_tmax_left:1]
    plot_x_right = np.array(right_x)
    plot_y_right = np.array(right_y)
    plot_tmin_right = base
    plot_tmax_right = tmax 
    xd_right = np.r_[plot_tmin_right:plot_tmax_right:1]
    bx = plt.axes()
    bx.plot(plot_x_left, plot_y_left, "o")
    bx.plot(plot_x_right, plot_y_right, "o")
    mean = -1.0
    if kind == 'HDD':
        if len(xd_right) > 0:
            '''
            meanlist = ave(plot_y_right, len(xd_right))
            mean = meanlist[0]
            '''
            meanlist = [base_load] * len(xd_right)
            bx.plot(xd_right, meanlist)
        if len(xd_left) > 0:
            '''
            if (mean > -1.0 and slope_l != 0 and not
                np.isnan(slope_l)): 
                plot_tmax_left = (meanlist[0] - intercept_l) / slope_l
                print ('modified base: {0}F'.format(plot_tmax_left))
                bx.annotate('break-even point: {0}F,\nbase load: {1}'.format(int(round(plot_tmax_left, 0)), round(mean, 1)), xy = (plot_tmax_left, mean), xytext = (plot_tmax_left, mean + 0.2), fontsize=15)
            '''
            xd_left = np.r_[plot_tmin_left:plot_tmax_left:1]
            bx.plot(xd_left, fit(xd_left, slope_l, intercept_l))
    else:
        if len(xd_left) > 0:
            meanlist = ave(plot_y_left, len(xd_left))
            mean = meanlist[0]
            bx.plot(xd_left, ave(plot_y_left, len(xd_left)))
        if len(xd_right) > 0:
            if (mean > -1.0 and slope_r != 0 and not
                np.isnan(slope_r)): 
                plot_tmin_right = (meanlist[0] - intercept_r) / slope_r
                print ('modified base: {0}F'.format(plot_tmin_right))
                bx.annotate('break-even point: {0}F,\nbase load: {1}'.format(int(round(plot_tmin_right, 0)), round(mean, 1)), xy = (plot_tmin_right, mean), xytext = (plot_tmin_right - 13, mean + 0.2), fontsize=15)
            xd_right = np.r_[plot_tmin_right:plot_tmax_right:1]
            bx.plot(xd_right, fit(xd_right, slope_r, intercept_r))
    plt.title('{0} - Temperature Plot'.format(title_dict[theme]))
    plt.suptitle('Building {0}, Station {1}'.format(b, s))
    plt.xlabel('Temperature Deg F')
    plt.ylabel(ylabel_dict[theme])
    P.savefig(os.getcwd() + '/plot_FY_weather/temp_energy/{2}/{0}_{1}.png'.format(b, s, theme), dpi = 150)
    plt.close()
    
def calculate_dd_energy_regression(kind, bs_pair):
    counter = 0
    bs = []
    ss = []
    slopes = []
    intercepts = []
    rs = []
    bases = []
    for (b, s) in bs_pair:
        print counter
        slope, intercept, r_value, basetemp = opt_lireg(b, s, kind)
        counter += 1
        bs.append(b)
        ss.append(s)
        slopes.append(slope)
        intercepts.append(intercept)
        rs.append(r_value)
        bases.append(int(basetemp[:2]))
    summary = pd.DataFrame({'Building Number': bs, 'Weather Station':
                            ss, 'Base Temperature': bases, 'k':
                            slopes, 'b': intercepts, 'r': rs})
    summary.to_csv(homedir + '{0}_regression.csv'.format(kind),
                   index=False)

# calculate savings for 'year', 'cutoff': r square cutoff
def calculate_savings(theme, kind, cutoff, year):
    df_reg = pd.read_csv(homedir + '{0}_regression.csv'.format(kind))
    df_reg['r2'] = df_reg['r'].map(lambda x: x * x)
    df_reg = df_reg[df_reg['r2'] >= cutoff]
    df_reg_idx = df_reg.set_index('Building Number')
    bs = df_reg['Building Number'].tolist()
    #print df_reg_idx.head()
    filelist = ['{0}dd_temp_eng/{1}_{2}_{3}.csv'.\
                format(homedir, kind, b, 
                       df_reg_idx.ix[b, 'Weather Station']) for b in bs]
    for f in filelist:
        df = pd.read_csv(f)
        filename = f[f.rfind('/') + 1:]
        b = filename[4: 12]
        s = filename[13: 17]
        print b, s
        df = df[df['year'] == year]
        slope = df_reg_idx.ix[b, 'k']
        intercept = df_reg_idx.ix[b, 'b']
        t_base = str(df_reg_idx.ix[b, 'Base Temperature']) + 'F'
        df = df[['Building Number', s, 'eui_elec', 'eui_gas', 'year',
                 'month', t_base]]
        print (slope, intercept, t_base)
        if kind == 'HDD':
            df['eui_gas_hat'] = df.apply(lambda r: slope * r[t_base] + intercept if r[t_base] > 0 else r['eui_gas'], axis=1)
        else:
            df['eui_elec_hat'] = df.apply(lambda r: slope * r[t_base] + intercept if r[t_base] > 0 else r['eui_elec'], axis=1)
        df.to_csv('{0}saving_{1}/{4}/{2}_{3}.csv'.format(homedir, year, b, s, theme), index=False)
    return
    
def plot_saving_two(theme, kind):
    sns.set_style("white")
    sns.set_context("talk", font_scale=1.5)
    filelist_1 = glob.glob('{0}saving_2014/{1}/*.csv'.format(homedir,
                                                             theme))
    filelist_2 = glob.glob('{0}saving_2015/{1}/*.csv'.format(homedir,
                                                             theme))
    df_summary = pd.read_csv(homedir +
                             '{0}_regression_fuel.csv'.format(kind))
    df_summary.set_index('Building Number', inplace=True)
    if kind == 'HDD':
        c1 = 'brown'
        c2 = 'lightsalmon'
        location = 'upper center'
        wrapwidth = 30
    else:
        c1 = 'navy'
        c2 = 'lightskyblue'
        location = 'lower center'
        wrapwidth = 99

    for (f1, f2) in zip(filelist_1, filelist_2):
        df_1 = pd.read_csv(f1)
        filename_1 = f1[f1.rfind('/') + 1:]
        b_1 = filename_1[:8]
        s_1 = filename_1[9: 13]
        k_1 = df_summary.ix[b_1, 'k']
        if k_1 == 0:
            continue
        r = df_summary.ix[b_1, 'r']
        r2 = round(r * r, 3)
        print (b_1, s_1)
        x_1 = df_1['month']
        y1_1 = df_1[theme]
        y2_1 = df_1[theme + '_hat']
        if y2_1.sum() != 0:
            save_percent_1 = int(round((y2_1.sum() - y1_1.sum()) /
                                       y2_1.sum() * 100, 0))
        else:
            save_percent_1 = 0
        df_2 = pd.read_csv(f2)
        filename_2 = f1[f2.rfind('/') + 1:]
        b_2 = filename_2[:8]
        s_2 = filename_2[9: 13]
        x_2 = df_2['month']
        y1_2 = df_2[theme]
        y2_2 = df_2[theme + '_hat']
        if y2_2.sum() != 0:
            save_percent_2 = int(round((y2_2.sum() - y1_2.sum()) /
                                       y2_2.sum() * 100, 0))
        else:
            save_percent_2 = 0

        fig, (ax_1, ax_2) = plt.subplots(2, 1, sharex=True,
                                         sharey=True)
        line1_1, = ax_1.plot(x_1, y1_1, c=c1, ls='-', lw=2, marker='o')
        line2_1, = ax_1.plot(x_1, y2_1, c=c2, ls='-', lw=2, marker='o')
        ax_1.fill_between(x_1, y1_1, y2_1, where=y2_1 >= y1_1,
                          facecolor='aquamarine', alpha=0.5,
                          interpolate=True)
        ax_1.fill_between(x_1, y1_1, y2_1, where=y2_1 < y1_1,
                          facecolor='orange', alpha=0.5,
                          interpolate=True)
        ax_1.legend([line1_1, line2_1], 
                    ['Actual {1} use in {0}'.format(2014, title_dict[theme]), '\n'.join(tw.wrap('{1} use given before 2013 habits but {0} weather'.format(2014, title_dict[theme]), wrapwidth))], loc=location)
        line1_2, = ax_2.plot(x_2, y1_2, c=c1, ls='-', lw=2, marker='o')
        line2_2, = ax_2.plot(x_2, y2_2, c=c2, ls='-', lw=2, marker='o')
        ax_2.fill_between(x_2, y1_2, y2_2, where=y2_2 >= y1_2,
                          facecolor='aquamarine', alpha=0.5,
                          interpolate=True)
        ax_2.fill_between(x_2, y1_2, y2_2, where=y2_2 < y1_2,
                          facecolor='orange', alpha=0.5,
                          interpolate=True)
        ax_2.legend([line1_2, line2_2], 
                    ['Actual {1} use in {0}'.format(2015, title_dict[theme]), '\n'.join(tw.wrap('{1} use given before 2013 habits but {0} weather'.format(2015, title_dict[theme]), wrapwidth))], loc=location)
        if save_percent_1 > 0:
            ax_1.set_title('{2} Savings Plot {0} vs before 2013, {1}% less, R^2: {3}'.format(2014, save_percent_1, title_dict[theme], r2))
        else:
            ax_1.set_title('{2} Savings Plot {0} vs before 2013, {1}% more R^2: {3}'.format(2014, abs(save_percent_1), title_dict[theme], r2))
        if save_percent_2 > 0:
            ax_2.set_title('{2} Savings Plot {0} vs before 2013, {1}% less R^2: {3}'.format(2015, save_percent_2, title_dict[theme], r2))
        else:
            ax_2.set_title('{2} Savings Plot {0} vs before 2013, {1}% more R^2: {3}'.format(2015, abs(save_percent_2), title_dict[theme], r2))
        plt.xticks(range(1, 13))
        xticklabels = [calendar.month_abbr[m] for m in range(1, 13)]
        plt.setp(ax_2, xticklabels=xticklabels)
        plt.xlim((1, 12))
        ylimit = max(max(y1_1.max(), y2_2.max()), max(y1_2.max(),
                                                      y2_2.max()))
        plt.ylim((0, ylimit * 1.1))
        plt.suptitle('Building {0}, Station {1}'.format(b_1, s_1))
        ax_1.set_ylabel('kBtu/sq.ft.')
        ax_2.set_ylabel('kBtu/sq.ft.')
        P.savefig(os.getcwd() + '/plot_FY_weather/saving/{2}/{0}_{1}.png'.format(b_1, s_1, theme), dpi = 150)
        plt.close()

def plot_saving(year, theme, kind):
    sns.set_style("white")
    sns.set_context("talk", font_scale=1.5)
    filelist = glob.glob('{0}saving_{1}/{2}/*.csv'.format(homedir, year, theme))
    df_summary = pd.read_csv(homedir +
                             '{0}_regression_fuel.csv'.format(kind))
    df_summary.set_index('Building Number', inplace=True)
    if kind == 'HDD':
        c1 = 'brown'
        c2 = 'lightsalmon'
    else:
        c1 = 'navy'
        c2 = 'lightskyblue'
    for f in filelist:
        df = pd.read_csv(f)
        filename = f[f.rfind('/') + 1:]
        b = filename[:8]
        s = filename[9: 13]
        r = df_summary.ix[b, 'r']
        r2 = round(r * r, 3)
        print (filename, b, s)
        x = df['month']
        y1 = df[theme]
        y2 = df[theme + '_hat']
        if y2.sum() != 0:
            save_percent = int(round((y2.sum() - y1.sum()) / y2.sum() *
                                    100, 0))
        else:
            save_percent = 0
        bx = plt.axes()
        line1, = bx.plot(x, y1, c=c1, ls='-', lw=2, marker='o')
        line2, = bx.plot(x, y2, c=c2, ls='-', lw=2, marker='o')
        bx.fill_between(x, y1, y2, where=y2 >= y1,
                         facecolor='aquamarine', alpha=0.5,
                         interpolate=True)
        bx.fill_between(x, y1, y2, where=y2 < y1, facecolor='orange',
                         alpha=0.5, interpolate=True)
        bx = plt.axes()
        if kind == 'HDD':
            location = 'upper left'
        else:
            location = 'lower left'
        plt.legend([line1, line2], 
                   ['Actual {1} use in {0}'.format(year, title_dict[theme]), 
                    '{1} use given before 2013 habits but {0} weather'.format(year, title_dict[theme])], 
                   loc=location)
        plt.xticks(range(1, 13))
        xticklabels = [calendar.month_abbr[m] for m in range(1, 13)]
        bx.set(xticklabels=xticklabels)
        plt.xlim((1, 12))
        plt.ylim((0, max(y1.max(), y2.max()) * 1.1))
        if save_percent > 0:
            plt.title('{2} Savings Plot {0} vs before 2013, {1}% less, R^2: {3}'.format(year, save_percent, title_dict[theme], r2))
        else:
            plt.title('{2} Savings Plot {0} vs before 2013, {1}% more, R^2: {3}'.format(year, abs(save_percent), title_dict[theme], r2))
        plt.suptitle('Building {0}, Station {1}'.format(b, s))
        #plt.xlabel('{0} Deg F'.format(kind))
        plt.ylabel(ylabel_dict[theme])
        P.savefig(os.getcwd() + '/plot_FY_weather/saving_{3}/{2}/{0}_{1}.png'.format(b, s, theme, year), dpi = 150)
        plt.close()

# join HDD_regression and CDD regression with 
# indicator_all for fuel type
def join_regression_indi():
    df_hdd = pd.read_csv(homedir + 'HDD_regression.csv')
    df_cdd = pd.read_csv(homedir + 'CDD_regression.csv')
    df_indi = pd.read_csv(os.getcwd() + \
                          '/csv_FY/filter_bit/fis/indicator_all_fuel.csv')
    # 'None (all electric?)' and 'Chilled Water' to be taken out for
    # electricity plot
    # 'Gas Only', the set used for gas plot
    cols = ['Building Number'] + [c for c in list(df_indi) if \
                                  ('None (all electric?)' in c) or \
                                  ('Chilled Water' in c) or \
                                  ('Gas Only' in c)]
    df_indi = df_indi[cols]
    print cols
    df_hdd_fuel = pd.merge(df_hdd, df_indi, on='Building Number',
                           how='inner')
    df_hdd_fuel['r2'] = df_hdd_fuel['r'] * df_hdd_fuel['r']
    df_cdd_fuel = pd.merge(df_cdd, df_indi, on='Building Number',
                           how='inner')
    df_cdd_fuel['r2'] = df_cdd_fuel['r'] * df_cdd_fuel['r']
    df_hdd_fuel.to_csv(homedir + 'HDD_regression_fuel.csv',
                       index=False)
    df_cdd_fuel.to_csv(homedir + 'CDD_regression_fuel.csv',
                       index=False)

# FIXME: not always set w, h right
def plot_stat_regression():
    sns.set_style("white")
    sns.set_palette("Set2")
    sns.set_context("talk", font_scale=1.0)
    my_dpi = 300

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    df = pd.read_csv(homedir + 'CDD_regression_fuel.csv')
    df = df[df['Chilled Water_10'] == 0]
    df = df[df['Chilled Water_12'] == 0]
    df = df[df['None (all electric?)_10'] == 0]
    df = df[df['None (all electric?)_12'] == 0]
    df2 = pd.read_csv(homedir + 'HDD_regression_fuel.csv')
    df2 = df2[df2['Gas Only_10'] == 1]
    df2 = df2[df2['Gas Only_12'] == 1]
    sns.boxplot(y='r2', data=df, ax=ax1)
    sns.boxplot(y='r2', data=df2, ax=ax2)
    ax1.set_ylabel('R square')
    ax1.set_title('Electricity - CDD regression distribution\n' + \
                  'No district chilled water no all electric building \n(n = {0})'.format(len(df)))
    ax2.set_title('Gas - HDD regression distribution\n' + \
                  'Gas Heating Only \n(n = {0})'.format(len(df2)))
    ax2.set_ylabel('')
    P.savefig(os.getcwd() + '/plot_FY_weather/summary/regression.png', dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi))
    plt.close()
    print 'end'

def plot_dd_energy_byyear(kind, theme, cutoff):
    sns.set_style("white")
    sns.set_context("talk", font_scale=1.0)
    df_reg = pd.read_csv(homedir + '{0}_regression.csv'.format(kind))
    df_reg['r2'] = df_reg['r'].map(lambda x: x * x)
    df_reg = df_reg[df_reg['r2'] >= cutoff]
    df_reg_idx = df_reg.set_index('Building Number')
    bs = df_reg['Building Number'].tolist()
    filelist = ['{0}dd_temp_eng/{1}_{2}_{3}.csv'.\
                format(homedir, kind, b, 
                       df_reg_idx.ix[b, 'Weather Station']) \
                for b in bs]
    if kind == 'HDD':
        colors = ['lightpink', 'deeppink', 'lightskyblue', 'navy']
    else:
        colors = ['lightsalmon', 'brown', 'aquamarine', 'teal']
    for f in filelist:
        df = pd.read_csv(f)
        if df[theme].sum() == 0:
            continue
        filename = f[f.rfind('/') + 1:]
        b = filename[4: 12]
        s = filename[13: 17]
        print b, s
        t_base = str(df_reg_idx.ix[b, 'Base Temperature']) + 'F'
        df = df[['Building Number', s, 'eui_elec', 'eui_gas', 'year',
                 'month', t_base]]
        df = df[df['year'] >= 2012]
        df['GSALink rollout'] = df['year'].map(lambda x: 'Before' if x < 2014 else 'After')
        if kind == 'CDD':
            sns.set_palette(sns.color_palette('Blues'))
        elif kind == 'HDD':
            df[t_base] = df[t_base] * (-1.0)
            sns.set_palette(sns.color_palette('Oranges'))
        g = sns.lmplot(x=t_base, y=theme, hue='year', data=df,
                       palette=sns.color_palette(colors),
                       fit_reg=True, size=4, aspect=1)
        plt.title('Building {0}, Station {1}, base {2}'.format(b, s, t_base))
        plt.xlabel('{0} Deg F'.format(kind))
        plt.ylabel(ylabel_dict[theme])
        if kind == 'HDD':
            plt.xlim((df[t_base].min(), 0))
        else:
            plt.xlim((0, df[t_base].max()))
        plt.ylim((0, df[theme].max() * 1.1))
        P.savefig(os.getcwd() + '/plot_FY_weather/{0}_{1}/{2}_{3}.png'.format(kind, theme, b, s), dpi = 300)
        plt.close()

def saving_summary(kind, theme):
    df_dd = pd.read_csv(homedir + '{0}_regression.csv'.format(kind))
    dfs = []
    for year in [2014, 2015]:
        filelist = glob.glob('{0}saving_{1}/{2}/*.csv'.format(homedir, year, theme))
        bs = []
        saves = []
        for f in filelist:
            filename = f[f.rfind('/') + 1:]
            b = filename[:8]
            s = filename[9: 13]
            print (b, s)
            df = pd.read_csv(f)
            y = df[theme]
            y_hat = df[theme + '_hat']
            if y_hat.sum() != 0:
                save_percent = (y_hat.sum() - y.sum()) / y_hat.sum() * 100
            else:
                save_percent = 0
            bs.append(b)
            saves.append(save_percent)
        d = {'Building Number': bs, 'Saving_{0}'.format(year): saves}
        df_year = pd.DataFrame(d)
        dfs.append(df_year)
    df_all = reduce(lambda x, y: pd.merge(x, y, how='left', on='Building Number'), [df_dd] + dfs)
    print df_all.head()
    df_all.to_csv(homedir + '{0}_saving_summary.csv'.format(kind),
                  index=False)

def process_gsalink():
    bs_pair = read_building_weather('building_station_lookup.csv',
                                    'Building Number', 'Weather Station')
    study_set = get_gsalink_set()
    bs_pair = [x for x in bs_pair if x[0] in study_set]
    cutoff = -0.1 # no cutoff limitation for R square
    '''
    for (b, s) in bs_pair:
        join_dd_temp_energy(b, s, 'CDD')
        join_dd_temp_energy(b, s, 'HDD')
    calculate_dd_energy_regression('HDD', bs_pair)
    calculate_dd_energy_regression('CDD', bs_pair)
    saving_summary('CDD', 'eui_gas')
    saving_summary('HDD', 'eui_elec')
    join_regression_indi()
    plot_stat_regression()
    for year in [2014, 2015]:
        calculate_savings('eui_elec', 'CDD', cutoff, year)
        calculate_savings('eui_gas', 'HDD', cutoff, year)
    for year in [2014, 2015]:
        plot_saving(year, 'eui_elec', 'CDD')
        plot_saving(year, 'eui_gas', 'HDD')
    plot_saving_two('eui_elec', 'CDD')
    plot_saving_two('eui_gas', 'HDD')
    plot_dd_energy_byyear('CDD', 'eui_elec', cutoff)
    plot_dd_energy_byyear('HDD', 'eui_gas', cutoff)
    '''

def main():
    # BOOKMARK regenerate weather
    process_weatherfile()
    # process_gsalink()

    #calculate('eui_gas', 'kernel')
    #calculate('eui_elec', 'kernel')
    #plot_building_temp()

    return
main()
