import os
import pandas as pd
import numpy as np
import glob
import read_fy_withcat as rd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as P
import matplotlib
import calendar

sns.set_palette(sns.color_palette('Set2', 4))
def get_limit():
    '''
    filelist = glob.glob(os.getcwd() + '/csv_FY/single_eui_cal/*.csv')
    dfs = [pd.read_csv(csv) for csv in filelist]
    df_all = pd.concat(dfs, ignore_index=False)
    print df_all.max()
    print df_all.quantile(0.95)
    df_all = df_all[['eui_elec', 'eui_gas', 'eui_oil', 'eui_water', 'eui']]
    df_stat = df_all.quantile(0.95)
    print df_stat
    print type(df_stat)
    d = df_stat.to_dict()
    print d

    for key in d:
        d[key] = np.ceil(d[key]/10) * 10
    print d
    m = max(d.values())
    for key in ['eui', 'eui_gas', 'eui_elec']:
        d[key] = m
    print d
    '''
    d = dict(zip(['eui_elec', 'eui_gas', 'eui_oil', 'eui_water', 'eui'],
                 [15, 15, 3, 6, 15]))
    return d

def plot_month_one(b_list, col, bd):
    dfs = [pd.read_csv(csv) for csv in b_list]
    dfs2 = [df[[col, 'month', 'year']] for df in dfs]
    df3 = (pd.concat(dfs2)).set_index(['month', 'year'])
    df_reshape = df3.unstack(level = -1)
    trend = df_reshape.plot()
    plt.xticks(range(1, 13))
    plt.yticks(range(0, 16, 3))
    xticklabels = [calendar.month_abbr[m] for m in range(1, 13)]
    trend.set(xticklabels=xticklabels)
    plt.xlim((1, 12))
    plt.title('{0}\nBuilding {1}'.format(title_dict[col], bd), fontsize=20, x = 0.5, y = 1)
    plt.ylim((0, ylimit[col]))
    plt.ylabel(ylabel_dict[col], fontsize=15)
    plt.xlabel('month', fontsize=15)
    trend.xaxis.set_label_coords(0.5, -0.08)
    plt.tick_params(axis='both', labelsize=15)
    #plt.show()
    P.savefig(os.getcwd() + '/plot_FY_cal/{0}/{1}.png'.format(col, bd), dpi = 75)
    plt.close()

def plot_month():
    filelist = glob.glob(os.getcwd() + '/csv_FY/single_eui_cal/*.csv')
    bd_set = set([f[f.find('_cal/') + 5:f.find('_cal/') + 13] for f in filelist])
    counter = 0
    for bd in list(bd_set):
        print counter
        print bd
        b_list = [f for f in filelist if bd in f]
        plot_month_one(b_list, 'eui', bd)
        plot_month_one(b_list, 'eui_elec', bd)
        plot_month_one(b_list, 'eui_gas', bd)
        plot_month_one(b_list, 'eui_oil', bd)
        plot_month_one(b_list, 'eui_water', bd)
        counter += 1

ylimit = get_limit()
title_dict = {'eui':'Not Weather Normalized Electricity + Gas Consumption',
              'eui_elec':'Not Weather Normalized Electricity Consumption',
              'eui_gas':'Not Weather Normalized Natural Gas Consumption',
              'eui_oil':'Not Weather Normalized Oil Consumption',
              'eui_water':'Not Weather Normalized Water Consumption'}

ylabel_dict = {'eui':'Electricity + Gas [kBtu/sq.ft]',
               'eui_elec':'Electricity [kBtu/sq.ft]',
               'eui_gas':'Natural Gas [kBtu/sq.ft]',
               'eui_oil':'Oil [Gallons/sq.ft]',
               'eui_water':'Water [Gallons/sq.ft]'}
def main():
    plot_month()
    return

main()
