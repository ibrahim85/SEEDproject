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

ylabel_dict = {'eui':'Electricity + Gas [kBtu/sq.ft]',
               'eui_elec':'Electricity [kBtu/sq.ft]',
               'eui_gas':'Natural Gas [kBtu/sq.ft]',
               'eui_oil':'Oil [Gallons/sq.ft]',
               'all': 'Electricity-Gas [kBtu/sq.ft]',
               'eui_water':'Water [Gallons/sq.ft]'}

title_dict = {'eui':'Electricity + Gas',
              'eui_elec':'Electricity',
              'eui_gas':'Natural Gas',
              'eui_oil':'Oil',
              'all': 'Combined Electricity and Gas',
              'eui_water':'Water'}

kind_dict = {'temp': 'Temperature', 'hdd': 'HDD', 'cdd': 'CDD', 'all': 'Combined'}

title_dict_2 = {'eui':'Original and Weather Normalized Electricity + Gas Consumption', 'eui_elec':'Original and Weather Normalized Electricity Consumption', 'eui_gas':'Original and Weather Normalized Natural Gas Consumption', 'eui_oil':'Original and Weather Normalized Oil Consumption', 'eui_water':'Original and Weather Normalized Water Consumption'}

xlabel_dict = {'temp': 'Monthly Mean Temperature, Deg F',
               'hdd': 'Monthly HDD, Deg F',
               'all': 'Monthly HDD(-)/CDD(+), Deg F',
               'cdd': 'Monthly CDD, Deg F'}

def excel2csv():
    filename = os.getcwd() + '/input/FY/weatherData.xlsx'
    df = pd.read_excel(filename, sheetname=0)
    df.to_csv(os.getcwd() + '/csv_FY/weather/weatherData.csv')

def check_data():
    '''
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData.csv')
    df.drop([0, 1], inplace=True)
    df.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_2.csv', index=False)
    '''
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_2.csv', nrows=27000, low_memory=False)
    print df['KBOS'].tail()
    cols = list(df.columns.values)

    df.replace(['No Data', 'Arc Off-line'], np.nan, inplace=True)
    df.replace(' ', np.nan, inplace=True)
    df.dropna(axis=1, how='any', inplace=True)
    cols = list(df.columns.values)
    assert('KDMH' not in df)
    df.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_nonan.csv',
              index=False)
    return

def get_mean_temp():
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_nonan.csv')
    df.set_index(pd.DatetimeIndex(df['Unnamed: 0']), inplace=True)
    df.resample('M', how = 'mean').to_csv(os.getcwd() + '/csv_FY/weather/weatherData_meanTemp.csv')

#source: http://energy.gov/sites/prod/files/2015/02/f20/Energy%20Intensity%20Baselining%20and%20Tracking%20Guidance.pdf
def get_HDD():
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_nonan.csv')
    df.set_index(pd.DatetimeIndex(df['Unnamed: 0']), inplace=True)
    df_day = df.resample('D', how = 'mean')
    #df_day.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_DayMeanTemp.csv')

    print df_day['KBOS'].head()
    for base in [40.0, 45.0, 50.0, 55.0, 57.0, 60.0, 65.0]:
        for col in df_day:
            df_day[col] = df_day[col].map(lambda x: 0 if x >= base else base - x)
        df_day.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_DayHDD_{0}F.csv'.format(int(base)))
        df_day.resample('M', how = 'sum').to_csv(os.getcwd() + '/csv_FY/weather/weatherData_HDD_{0}F.csv'.format(int(base)))

def get_DD_itg(base, theme):
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_nonan.csv')
    df.set_index(pd.DatetimeIndex(df['Unnamed: 0']), inplace=True)
    df_hour = df.resample('H', how = 'mean')
    #df_day.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_DayMeanTemp.csv')

    for col in df_hour:
        if theme == 'HDD':
            df_hour[col] = df_hour[col].map(lambda x: 0 if x >= base else base - x)
        else:
            df_hour[col] = df_hour[col].map(lambda x: 0 if x <= base else x - base)
    df_day = df_hour.resample('D', how = 'mean')
    print 'base temperature: {0}'.format(base)
    print df_day['KBOS'].head()
    df_day.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_Day{1}_itg_{0}F.csv'.format(int(base), theme))
    df_month = df_day.resample('M', how = 'sum').to_csv(os.getcwd() + '/csv_FY/weather/weatherData_{1}_itg_{0}F.csv'.format(int(base), theme))

def get_CDH():
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_nonan.csv')
    df.set_index(pd.DatetimeIndex(df['Unnamed: 0']), inplace=True)
    for base in [60.0]:
        for col in df:
            if col != 'Unnamed: 0':
                df[col] = df[col].map(lambda x: float(x))
                df[col] = df[col].map(lambda x: 0 if (x) <= base else (x) - base)
                print df[col].head()
        df_out = df.resample('M', how = 'sum')
        df_out.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_CDH_{0}F.csv'.format(int(base)))

def get_CDD_itg(base):
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_nonan.csv')
    df.set_index(pd.DatetimeIndex(df['Unnamed: 0']), inplace=True)
    df_hour = df.resample('H', how = 'mean')
    #df_day.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_DayMeanTemp.csv')

    for col in df_hour:
        df_hour[col] = df_day[col].map(lambda x: 0 if x <= base else x - base)
    df_day.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_DayCDD_{0}F.csv'.format(int(base)))
    df_day.resample('M', how = 'sum').to_csv(os.getcwd() + '/csv_FY/weather/weatherData_CDD_{0}F.csv'.format(int(base)))

def get_CDD():
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_nonan.csv')
    df.set_index(pd.DatetimeIndex(df['Unnamed: 0']), inplace=True)
    df_day = df.resample('D', how = 'mean')
    #df_day.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_DayMeanTemp.csv')

    print df_day['KBOS'].head()
    for base in [45.0, 50.0, 55.0, 57.0, 60.0, 65.0, 70.0, 72.0]:
        for col in df_day:
            df_day[col] = df_day[col].map(lambda x: 0 if x <= base else x - base)
        df_day.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_DayCDD_{0}F.csv'.format(int(base)))
        df_day.resample('M', how = 'sum').to_csv(os.getcwd() + '/csv_FY/weather/weatherData_CDD_{0}F.csv'.format(int(base)))

def read_building_weather():
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/building_station.csv')
    bs_pair = zip(df['Building ID'].tolist(), df['Station'].tolist())
    return bs_pair

def read_energy(b):
    filelist = glob.glob(os.getcwd() + '/csv_FY/single_eui_cal/{0}*.csv'.format(b))
    dfs = [pd.read_csv(csv) for csv in filelist]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.sort(columns=['year', 'month'], inplace=True)
    return df_all

# read temperature record from Sep. 2012 to Sep. 2015
def read_temperature():
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_meanTemp.csv')
    df.drop(0, axis=0, inplace=True)
    return df

# read icao cade (4-alphabetical char) of weather station
def read_icao():
    names = ['Block Number', 'Station Number', 'ICAO Location', 'Indicator',
             'Place Name', 'State', 'Country Name', 'WMO Region',
             'Station Latitude', 'Station Longitude', 'Upper Air Latitude',
             'Upper Air Longitude', 'Station Elevation (Ha)',
             'Upper Air Elevation (Hp)', 'RBSN indicator']
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/nsd_bbsss.txt', sep=';',
                     header=None, names=names)

    df['WMO ID'] = df.apply(lambda row: str(row['Block Number']).zfill(2) + str(row['Station Number']).zfill(3), axis=1)
    df = df[['WMO ID', 'ICAO Location']]
    return df

def read_ghcnd():
    names = ['ID', 'LATITUDE', 'LONGITUD', 'ELEVATION', 'STATE', 'NAME',
             'GSN FLAG', 'HCN/CRN FLAG', 'WMO ID']
    filename = os.getcwd() + '/csv_FY/weather/ghcnd-stations.txt'
    outfile = os.getcwd() + '/csv_FY/weather/ghcnd-stations-delim.txt'
    with open (filename, 'r') as rd:
        lines = rd.readlines()
    with open (outfile ,'w+') as wt:
        for line in lines:
            line_list = list(line)
            for i in [11, 20, 30, 37, 40, 71, 75, 79]:
                line_list[i] = ','
            new_line = ''.join(line_list)
            wt.write(new_line)
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/ghcnd-stations-delim.txt',
                     header=None, names=names)
    df = df[['ID', 'WMO ID']]
    return df

def read_ncdc():
    import calendar
    names = ['ID'] + [calendar.month_abbr[i] for i in range(1, 13)]
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/mly-tavg-normal.txt',
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
    #df_ghcnd.to_csv(os.getcwd() + '/csv_FY/weather/ghcnd.csv', index=False)
    df_merge = pd.merge(df_icao, df_ghcnd, on='WMO ID', how='left')
    df_merge = df_merge[df_merge['ICAO Location'] != '----']
    #df_merge.to_csv(os.getcwd() + '/csv_FY/weather/icao_ghcnd.csv',
    #                index=False)
    df_temp = read_ncdc()
    df_all = pd.merge(df_merge, df_temp, on='ID', how='left')
    #df_all.to_csv(os.getcwd() + '/csv_FY/weather/icao_ghcnd_ncdc.csv',
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

#plot_energy_temp_byyear_2015('eui_elec')
#plot_energy_temp_byyear_2015('eui_gas')

#ld: line or dot
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
            sns.lmplot(x=kind, y=theme, hue='Fiscal Year', data=df, fit_reg=True)
            x = np.array(df[kind])
            y = np.array(df[theme])
            t_min = df[kind].min()
            t_max = df[kind].max()
            xd = np.r_[t_min:t_max:1]
            k1 = smooth.NonParamRegression(x, y, method=npr_methods.LocalPolynomialKernel(q=1))
            plt.plot(xd, k1(xd), '-', color=sns.color_palette('Set2')[5])
            plt.xlabel(xlabel_dict[kind], fontsize=12)
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
            df['temp'] = df['temp'] - 65.0
            df['temp_dd'] = df.apply(lambda r: r['hdd'] if r['temp'] < 0 else r['cdd'], axis=1)
            df_elec = df
            if remove0:
                df_elec = df_elec[df_elec['cdd'] >= 10]
            df_elec['dd'] = df_elec['cdd']
            df_elec['eui_plot'] = df_elec['eui_elec']
            df_elec['kind'] = 'Electricity'
            df_elec = df_elec[['dd', 'eui_plot', 'kind', 'Fiscal Year']]
            df_gas = df
            if remove0:
                df_gas = df_gas[df_gas['hdd'] <= -10]
            df_gas['dd'] = df_gas['hdd']
            df_gas['eui_plot'] = df_gas['eui_gas']
            df_gas['kind'] = 'Gas'
            df_gas = df_gas[['dd', 'eui_plot', 'kind', 'Fiscal Year']]
            df_temp = df
            df_temp['dd'] = df_temp['temp_dd']
            df_temp['eui_plot'] = df_temp['eui']
            df_temp['kind'] = 'Combined'
            df_temp = df_temp[['dd', 'eui_plot', 'kind', 'Fiscal Year']]
            df_all = pd.concat([df_elec, df_gas, df_temp], ignore_index=True)
            g = sns.lmplot(x='dd', y='eui_plot', data=df_all, col='Fiscal Year', hue = 'kind', fit_reg=True, truncate=True)
            #plt.xlabel(xlabel_dict[kind], fontsize=12)
            g = g.set_axis_labels(xlabel_dict[kind],
                                  ylabel_dict[kind])
    plt.ylim((0, 10))
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
            P.savefig(os.getcwd() + '/plot_FY_weather/eui_{3}/{2}_byyear_dot_noreg/{0}_{1}.png'.format(b, s, theme, kind), dpi = 75)
        else:
            P.savefig(os.getcwd() + '/plot_FY_weather/eui_{2}/{2}_byyear_dot/{0}_{1}.png'.format(b, s, kind), dpi = 75)
    plt.close()

def regression_hdd(t, s, df, theme):
    df_temp = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_HDD_{0}F.csv'.format(t))
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

# TRY !!
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
    plt.title('{0}\nBuilding {1}'.format(title_dict_2[theme], b), fontsize=15, x = 0.5, y = 1)
    plt.xlim((1, 12))
    plt.xlabel('Month', fontsize=12)
    plt.xticks(range(1, 13), [calendar.month_abbr[m] for m in range(1, 13)])
    plt.ylabel(ylabel_dict[theme], fontsize=12)
    P.savefig(os.getcwd() + '/plot_FY_weather/{2}_ori_norm/{0}_{1}.png'.format(b, s, theme), dpi=150)
    plt.close()

title_dict_3 = {'eui':'Weather Normalized Electricity + Gas Consumption', 'eui_elec':'Weather Normalized Electricity Consumption', 'eui_gas':'Weather Normalized Natural Gas Consumption', 'eui_oil':'Weather Normalized Oil Consumption', 'eui_water':'Weather Normalized Water Consumption'}

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

def calculate(theme, method):
    bs_pair = read_building_weather()
    df_temperature = read_temperature()
    df_temp_norm = read_temp_norm()
    df_temp_norm.drop(['WMO ID', 'ID'], axis=1, inplace=True)
    df_hdd_65 = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_HDD_itg_65F.csv')
    df_hdd_65.drop(0, axis=0, inplace=True)
    df_cdh_65 = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_CDD_itg_65F.csv')
    df_cdh_65.drop(0, axis=0, inplace=True)

    '''
    t_norm = df_temp_norm[df_temp_norm['ICAO Location'] == s]
    print len(t_norm)
    if len(t_norm) == 0:
        print s
        continue
    t_norm_list = list(list(t_norm.itertuples())[0])[2:]
    '''
    for b, s in bs_pair:
        print (b, s)
        df_energy = read_energy(b)[['Fiscal Year', 'year', 'month', 'eui_elec', 'eui_gas', 'eui']]
        df_t = df_temperature[[s]]
        df_h = df_hdd_65[[s]]
        df_c = df_cdh_65[[s]]
        #plot_energy_temp(df_energy, df_t, theme, b, s)
        # plot with mean temp
        #plot_energy_temp_byyear(df_energy, df_t, theme, b, s, 'dot')
        '''
        plot_energy_temp_byyear(df_energy, df_t, df_h, df_c, theme, b, s,
                'dot', 'hdd', True)
        plot_energy_temp_byyear(df_energy, df_t, df_h, df_c, theme, b, s,
                'dot', 'cdd', True)
        plot_energy_temp_byyear(df_energy, df_t, df_h, df_c, theme,
                                b, s, 'line', 'all', True)
        '''
        plot_energy_temp_byyear(df_energy, df_t, df_h, df_c, theme,
                                b, s, 'dot', 'all', True)
        '''
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

def plot_building_degreeday():
    b = 'AZ0000FF'
    s = 'KTUS'
    filelist = glob.glob(os.getcwd() + '/csv_FY/testWeather/{0}*.csv'.format(b))
    dfs = [pd.read_csv(csv) for csv in filelist]
    col = 'eui_gas'
    dfs2 = [df[[col, 'month', 'year']] for df in dfs]
    df3 = (pd.concat(dfs2))
    hdd = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_HDD.csv')
    hdd['year'] = hdd['Unnamed: 0'].map(lambda x: float(x[:4]))
    hdd['month'] = hdd['Unnamed: 0'].map(lambda x: float(x[5:7]))
    hdd.set_index(pd.DatetimeIndex(hdd['Unnamed: 0']), inplace=True)
    hdd = hdd[[s, 'month', 'year']]
    hdd.info()
    joint = pd.merge(df3, hdd, on = ['year', 'month'], how = 'inner')
    joint.to_csv(os.getcwd() + '/csv_FY/testWeather/test.csv', index=False)
    joint_li = joint
    joint_li = joint_li[joint_li['year'] > 2012]
    joint_li = joint_li[joint_li['year'] < 2015]
    print 'len of joint{0}'.format(joint)

    '''
    def r2(x, y):
        return stats.pearsonr(x, y)[0] ** 2
    '''
    # calculate
    slope, intercept, r_value, p_value, std_err = stats.linregress(joint_li[s],joint_li[col])
    print '(slope, intercept, r_value, p_value, std_err)'
    plt.title('y = {0}x + {1}'.format(slope, intercept))
    print slope, intercept, r_value, p_value, std_err
    #sns.jointplot(s, col, kind="reg", stat_func=r2, data=joint)
    #sns.jointplot(s, col, kind="reg", data=joint, x_jitter=0.2, y_jitter=0.2)
    sns.regplot(s, col, data=joint)
    plt.xlim((0 - 10, hdd[s].max() + 10))
    plt.ylim((0, joint[col].max() + 0.1))
    plt.show()
    plt.close()

    #joint['predict'] = joint['HDD']

    '''
    temp = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_meanTemp.csv')
    temp['year'] = temp['Unnamed: 0'].map(lambda x: float(x[:4]))
    temp['month'] = temp['Unnamed: 0'].map(lambda x: float(x[5:7]))
    temp.set_index(pd.DatetimeIndex(temp['Unnamed: 0']), inplace=True)
    temp = temp[[s, 'month', 'year']]
    joint2 = pd.merge(df3, temp, on = ['year', 'month'], how = 'inner')
    joint2.to_csv(os.getcwd() + '/csv_FY/testWeather/test_temp.csv', index=False)

    sns.lmplot(s, col, data=joint, col='year', fit_reg=False)
    plt.xlim((0, joint2[s].max()))
    plt.ylim((0, joint2[col].max()))
    plt.show()
    plt.close()
    '''
# BOOKMARK
def get_good_building():
    print 'not implemented'
    df = pd.read_csv(os.getcwd() + '/csv_FY/filter_bit/indicator_all.csv')
    good_office = df[df['office' == 1]]
    good_station = set(list(pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_meanTemp.csv').columns.values()))
    station_info = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherStation.csv')
    station_info = station_info[['Building Number', 'Weather Station']]

def plot_building_temp():
    sns.set_context("paper", font_scale=1.5)
    b = 'AZ0000FF'
    s = 'KTUS'
    filelist = glob.glob(os.getcwd() + '/csv_FY/testWeather/{0}*.csv'.format(b))
    dfs = [pd.read_csv(csv) for csv in filelist]
    col = 'eui_gas'
    dfs2 = [df[[col, 'month', 'year']] for df in dfs]
    df3 = (pd.concat(dfs2))

    temp = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_meanTemp.csv')
    temp['year'] = temp['Unnamed: 0'].map(lambda x: float(x[:4]))
    temp['month'] = temp['Unnamed: 0'].map(lambda x: float(x[5:7]))
    temp.set_index(pd.DatetimeIndex(temp['Unnamed: 0']), inplace=True)
    temp = temp[[s, 'month', 'year']]
    joint2 = pd.merge(df3, temp, on = ['year', 'month'], how = 'inner')
    joint2.to_csv(os.getcwd() + '/csv_FY/testWeather/test_temp.csv', index=False)

    sns.lmplot(s, col, data=joint2, col='year', fit_reg=False)
    plt.xlim((joint2[s].min() - 10, joint2[s].max() + 10))
    plt.ylim((0, joint2[col].max() + 0.1))
    P.savefig(os.getcwd() + '/csv_FY/testWeather/plot/scatter_temp_byyear.png', dpi=150)
    plt.close()

    joint2 = joint2[(2012 < joint2['year']) & (joint2['year'] < 2015)]
    sns.regplot(s, col, data=joint2, fit_reg=False)
    plt.xlim((joint2[s].min() - 10, joint2[s].max() + 10))
    plt.ylim((0, joint2[col].max() + 0.1))
    P.savefig(os.getcwd() + '/csv_FY/testWeather/plot/scatter_temp_1314.png', dpi=150)
    plt.close()
    # BOOKMARK

def main():
    #excel2csv()
    #check_data()
    #get_mean_temp()
    #get_HDD()

    '''
    for base in [40.0, 45.0, 50.0, 55.0, 57.0, 60.0, 65.0]:
        get_DD_itg(base, 'HDD')
    for base in [45.0, 50.0, 55.0, 57.0, 60.0, 65.0, 70.0, 72.0]:
        get_DD_itg(base, 'CDD')
    '''
    #get_CDD()
    #get_CDH()
    #for theme in ['eui_elec', 'eui_gas']:
    for theme in ['eui_gas']:
    #for theme in ['eui_elec']:
        calculate(theme, 'kernel')
    #plot_building_temp()
    return
main()
