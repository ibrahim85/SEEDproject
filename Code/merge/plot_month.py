import os
import pandas as pd
import glob
import read_fy_withcat as rd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as P
import matplotlib
import calendar

sns.set_palette(sns.color_palette('Set2', 3))

def plot_month_one(b_list, col, bd):
    dfs = [pd.read_csv(csv) for csv in b_list]
    dfs2 = [df[[col, 'Fiscal Month', 'Fiscal Year']] for df in dfs]
    df3 = (pd.concat(dfs2)).set_index(['Fiscal Month', 'Fiscal Year'])
    df_reshape = df3.unstack(level = -1)
    #df_reshape.to_csv(os.getcwd() + '/csv_FY/for_plot/test.csv', index=False)
    trend = df_reshape.plot()
    plt.xlim((1, 12))
    plt.title('{0} of Building {1}'.format(col.upper(), bd), fontsize=20, x = 0.5, y = 1.02)
    plt.ylim((0, 10))
    xticklabels = [calendar.month_abbr[m] for m in range(1, 13)]
    trend.set(xticklabels = xticklabels)
    plt.ylabel(col.upper(), fontsize=15)
    plt.xlabel('Fiscal Month', fontsize=15)
    trend.xaxis.set_label_coords(0.5, -0.08)
    plt.tick_params(axis='both', labelsize=15)
    #plt.show()
    P.savefig(os.getcwd() + '/plot_FY/{0}/{1}.png'.format(col, bd), dpi = 75)
    plt.close()

def plot_month():
    filelist = glob.glob(os.getcwd() + '/csv_FY/single_eui/*.csv')
    bd_set = set([f[f.find('_eui/') + 5:f.find('_eui/') + 13] for f in filelist])
    counter = 0
    for bd in list(bd_set)[:1]:
        print counter
        print bd
        b_list = [f for f in filelist if bd in f]
        plot_month_one(b_list, 'eui', bd)
        plot_month_one(b_list, 'eui_elec', bd)
        plot_month_one(b_list, 'eui_gas', bd)
        plot_month_one(b_list, 'eui_oil', bd)
        plot_month_one(b_list, 'eui_water', bd)
        counter += 1

def main():
    plot_month()

main()
