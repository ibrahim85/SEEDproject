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
               'eui_water':'Water [Gallons/sq.ft]'}

title_dict = {'eui':'Electricity + Gas',
              'eui_elec':'Electricity',
              'eui_gas':'Natural Gas',
              'eui_oil':'Oil',
              'eui_water':'Water'}

title_dict_2 = {'eui':'Original and Weather Normalized Electricity + Gas Consumption', 'eui_elec':'Original and Weather Normalized Electricity Consumption', 'eui_gas':'Original and Weather Normalized Natural Gas Consumption', 'eui_oil':'Original and Weather Normalized Oil Consumption', 'eui_water':'Original and Weather Normalized Water Consumption'}

def excel2csv():
    filename = os.getcwd() + '/input/FY/weatherData.xlsx'
    df = pd.read_excel(filename, sheetname=0)
    df.to_csv(os.getcwd() + '/csv_FY/weather/weatherData.csv')

# check weather file
def check_data():
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData.csv')
    df.drop([0, 1], inplace=True)
    df.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_2.csv', index=False)
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_2.csv', nrows=27000)
    print df['KBOS'].tail()
    cols = list(df.columns.values)
    print len(cols)

    df.replace(['No Data', 'Arc Off-line'], np.nan, inplace=True)
    df.dropna(axis=1, how='any', inplace=True)
    cols = list(df.columns.values)
    print len(cols)
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
    df_day.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_DayMeanTemp.csv')

    for base in [40.0, 45.0, 50.0, 55.0, 57.0, 60.0, 65.0]:
        for col in df_day:
            df_day[col] = df_day[col].map(lambda x: 0 if x >= base else base - x)
        df_day.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_DayHDD_{0}F.csv'.format(int(base)))
        df_day.resample('M', how = 'sum').to_csv(os.getcwd() + '/csv_FY/weather/weatherData_HDD_{0}F.csv'.format(int(base)))

def get_CDD():
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_nonan.csv')
    df.set_index(pd.DatetimeIndex(df['Unnamed: 0']), inplace=True)
    df_day = df.resample('D', how = 'mean')
    df_day.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_DayMeanTemp.csv')

    for base in [45.0, 50.0, 55.0, 57.0, 60.0, 65.0, 70.0, 72.0]:
        for col in df_day:
            df_day[col] = df_day[col].map(lambda x: 0 if x <= base else x - base)
        df_day.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_DayCDD_{0}F.csv'.format(int(base)))
        df_day.resample('M', how = 'sum').to_csv(os.getcwd() + '/csv_FY/weather/weatherData_CDD_{0}F.csv'.format(int(base)))

# reading building, station lookup table
def read_building_weather():
    df = pd.read_csv(os.getcwd() + '/csv_FY/weather/building_station.csv')
    bs_pair = zip(df['Building ID'].tolist(), df['Station'].tolist())
    return bs_pair

# read monthly energy for building b
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

# read ncdc file
def read_ncdc():
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

# plot energy vs temperature
def plot_energy_temp(df_energy, df_temp, theme, b, s):
    df = pd.DataFrame({'energy': df_energy[theme], 'temp': df_temp[s]})
    sns.regplot('temp', 'energy', data=df, fit_reg=False)
    P.savefig(os.getcwd() + '/plot_FY_weather/{2}/{0}_{1}.png'.format(b, s, theme), dpi = 150)
    plt.title('Temperature-{0} plot: {1}, {2}'.format(theme, b, s))
    plt.close()
    return

# plot energy vs temperature of year 2015
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
    plt.xlabel('Temperature / F', fontsize=12)
    plt.ylabel(ylabel_dict[theme], fontsize=12)
    plt.legend(lines, labels, bbox_to_anchor=(0.2, 1), prop={'size':6})
    plt.ylim((0, 9))
    P.savefig(os.getcwd() + '/plot_FY_weather/27building_{0}_2015_trunc.png'.format(theme), dpi = 150)
    #P.savefig(os.getcwd() + '/plot_FY_weather/27building_{0}_2015.png'.format(theme), dpi = 150)
    plt.close()
    return

#plot_energy_temp_byyear_2015('eui_elec')
#plot_energy_temp_byyear_2015('eui_gas')

# plot energy vs temperature group by year
def plot_energy_temp_byyear(df_energy, df_temp, theme, b, s):
    sns.set_palette(sns.color_palette('Set2', 4))
    df = df_energy
    df['temperature'] = df_temp[s].tolist()
    df.to_csv(os.getcwd() + '/csv_FY/energy_temperature_select/{0}_{1}_{2}.csv'.format(b, s, title_dict[theme]), index=False)
    gr = df.groupby('Fiscal Year')
    lines = []
    for name, group in gr:
        group.sort(['temperature', theme], inplace=True)
        group = group[['temperature', theme]]
        line, = plt.plot(group['temperature'], group[theme])
        lines.append(line)
        print 'Building: {0}, year: {1}, {2} {3} [kbtu/sq.ft.]'.format(b, int(name), round(group[theme].sum(), 2), title_dict[theme])

    plt.title('Temperature-{0} plot: Building {1}, Station {2}'.format(title_dict[theme], b, s))
    plt.legend(lines, ['2013', '2014', '2015'], bbox_to_anchor=(1, 1))
    plt.xlim((20, 80))
    plt.ylim((0, 8))
    P.savefig(os.getcwd() + '/plot_FY_weather/{2}_byyear/{0}_{1}.png'.format(b, s, theme), dpi = 150)
    plt.close()
    return

# method 1: compute regression of energy vs. hdd
def regression_hdd(t, s, df, theme):
    df_temp = pd.read_csv(os.getcwd() + '/csv_FY/weather/weatherData_HDD_{0}F.csv'.format(t))
    df_temp.drop(0, inplace=True)
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_temp[s], df[theme])
    return (slope, intercept, r_value * r_value, t)

# method 2: piecewise regression of energy vs. temperature using piecewise
# linear regression
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

# method 3: regression of energy vs. temperature using local kernel linear regression
def regression_gas_kernel(b, s, df, theme):
    x = np.array(df['temperature'])
    y = np.array(df[theme])
    t_min = df['temperature'].min()
    t_max = df['temperature'].max()
    xd = np.r_[t_min:t_max:1]
    k1 = smooth.NonParamRegression(x, y, method=npr_methods.LocalPolynomialKernel(q=1))
    plt.plot(x, y, "o")
    plt.plot(xd, k1(xd))
    plt.xlabel('Temperature', fontsize=12)
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
        ori,  = plt.plot(group[theme])
        norm, = plt.plot(group['e_norm'])
        lines.append(ori)
        lines.append(norm)
    plt.legend(lines, ['2012_ori', '2012_norm', '2013_ori', '2013_norm', '2014_ori', '2014_norm', '2015_ori', '2015_norm'], bbox_to_anchor=(0.2, 1))
    plt.title('{0}\nBuilding {1}'.format(title_dict_2[theme], b), fontsize=15, x = 0.5, y = 1)
    plt.xlim((0, 11))
    plt.xlabel('Month', fontsize=12)
    plt.xticks(range(0, 12), [calendar.month_abbr[m] for m in range(1, 13)])
    plt.ylabel(ylabel_dict[theme], fontsize=12)
    P.savefig(os.getcwd() + '/plot_FY_weather/eui_gas_ori_norm/{0}_{1}.png'.format(b, s), dpi=150)
    plt.close()

title_dict_3 = {'eui':'Weather Normalized Electricity + Gas Consumption', 'eui_elec':'Weather Normalized Electricity Consumption', 'eui_gas':'Weather Normalized Natural Gas Consumption', 'eui_oil':'Weather Normalized Oil Consumption', 'eui_water':'Weather Normalized Water Consumption'}

def plot_normal_only(df, theme, b, s):
    sns.set_palette(sns.color_palette('Set2', 4))
    gr = df.groupby('year')
    lines = []
    for name, group in gr:
        norm, = plt.plot(group['e_norm'])
        lines.append(norm)
    plt.legend(lines, ['2012_norm', '2013_norm', '2014_norm', '2015_norm'], bbox_to_anchor=(0.2, 1))
    plt.title('{0}\nBuilding {1}'.format(title_dict_3[theme], b), fontsize=15, x = 0.5, y = 1)
    plt.xlim((0, 11))
    plt.xlabel('Month', fontsize=12)
    plt.xticks(range(0, 12), [calendar.month_abbr[m] for m in range(1, 13)])
    plt.ylabel(ylabel_dict[theme], fontsize=12)
    P.savefig(os.getcwd() + '/plot_FY_weather/eui_gas_norm/{0}_{1}.png'.format(b, s), dpi=150)
    plt.close()

def calculate(theme, method):
    bs_pair = read_building_weather()
    df_temperature = read_temperature()
    df_temp_norm = read_temp_norm()
    df_temp_norm.drop(['WMO ID', 'ID'], axis=1, inplace=True)

    for b, s in bs_pair:
        print (b, s)
        t_norm = df_temp_norm[df_temp_norm['ICAO Location'] == s]
        if len(t_norm) == 0:
            print s
            continue
        t_norm_list = list(list(t_norm.itertuples())[0])[2:]
        df_energy = read_energy(b)[['Fiscal Year', 'year', 'month', theme]]
        df_t = df_temperature[[s]]
        #plot_energy_temp(df_energy, df_t, theme, b, s)
        plot_energy_temp_byyear(df_energy, df_t, theme, b, s)
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
                df['e_norm_predict'] = reg(np.array(t_norm_list * 3))
                df['e_cur_predict'] = reg(df['temperature'])
                df['e_norm'] = df.apply(lambda r: max(0, r['e_norm_predict']/r['e_cur_predict'] * r[theme]), axis=1)
                plot_normal(df, theme, b, s)
                plot_normal_only(df, theme, b, s)

def main():
    #excel2csv()
    #check_data()
    #get_mean_temp()
    #get_HDD()
    #get_CDD()
    '''
    for theme in ['eui_elec', 'eui_gas']:
    #for theme in ['eui_gas']:
        calculate(theme, 'kernel')
    '''
    #plot_building_temp()
    return

main()
