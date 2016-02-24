import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import pylab as P
import matplotlib.pyplot as plt

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
    #df_day.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_DayMeanTemp.csv')

    print df_day['KBOS'].head()
    for base in [40.0, 45.0, 50.0, 55.0, 57.0, 60.0, 65.0]:
        for col in df_day:
            df_day[col] = df_day[col].map(lambda x: 0 if x >= base else base - x)
        df_day.to_csv(os.getcwd() + '/csv_FY/weather/weatherData_DayHDD_{0}F.csv'.format(int(base)))
        df_day.resample('M', how = 'sum').to_csv(os.getcwd() + '/csv_FY/weather/weatherData_HDD_{0}F.csv'.format(int(base)))

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

    from scipy import stats
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
    #get_CDD()
    #plot_building_temp()
    return
main()
