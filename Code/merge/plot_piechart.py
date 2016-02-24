import os
import pandas as pd
import numpy as np
import glob
import read_fy_withcat as rd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as P
import matplotlib
import operator

sns.set_palette(sns.color_palette('Set2', 4))
def read_cat():
    filelist_14 = glob.glob(os.getcwd() + '/csv_FY/sep/FY14*.csv')
    filelist_15 = glob.glob(os.getcwd() + '/csv_FY/sep/FY15*.csv')
    both = filelist_14 + filelist_15
    lists = [filelist_14, filelist_15, both]
    namelist = ['14', '15', 'both']
    for i in range(3):
        dfs = [pd.read_csv(csv) for csv in lists[i]]
        dfs2 = [df[['Building Number', 'Cat']] for df in dfs]
        df3 = pd.concat(dfs2, ignore_index=True)
        df3.drop_duplicates(cols='Building Number', inplace=True)
        df3.to_csv(os.getcwd() + '/csv_FY/cat_type/cat_{0}.csv'.format(namelist[i]), index=False)

def plot():
    filelist = glob.glob(os.getcwd() + '/csv_FY/cat_type/*.csv')
    d = {'14': '2014', '15': '2015', 'both': '2014 and 2015'}
    for csv in filelist:
        df = pd.read_csv(csv)
        s = 'type/cat_'
        sufname = csv[csv.find(s) + len(s): -4]
        total = df['Cat'].value_counts().sum()
        percent = df['Cat'].value_counts()/df['Cat'].value_counts().sum() * 100
        d_percent = percent.to_dict()
        labels = d_percent.keys()
        fracs = d_percent.values()

        colors = sns.color_palette('Set3', len(labels))
        plt.pie(fracs, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')
        plt.title('Pie chart of category for {0}: {1} buildings'.format(d[sufname], total), x=0.5, y=1.05, fontsize=15)
        P.savefig(os.getcwd() + '/plot_FY_pie/{0}.png'.format(sufname), dpi = 75)
        plt.close()

def read_type():
    df_type = pd.read_csv(os.getcwd() + '/csv/all_column/sheet-0-all_col.csv')
    df_type = df_type[['Property Name', 'Self-Selected Primary Function']]
    df_type['Property Name'] = df_type['Property Name'].map(lambda x: x.partition(' ')[0][:8])
    df_type.drop_duplicates(cols='Property Name', inplace=True)
    print len(df_type)
    df_bd = pd.read_csv(os.getcwd() + '/csv_FY/filter_bit/indicator_all.csv')
    print len(df_bd)
    df_all = pd.merge(df_bd, df_type, left_on = 'Building Number',
                      right_on = 'Property Name', how = 'left')
    print len(df_all)
    df_all.fillna({'Self-Selected Primary Function': 'No Data'}, inplace=True)
    df_all.drop('Property Name', inplace=True, axis=1)
    df_all.to_csv(os.getcwd() + '/csv_FY/filter_bit/indicator_all_type.csv',
                  index=False)

def plot_type():
    sns.mpl.rc("figure", figsize=(10,5.5))
    df = pd.read_csv(os.getcwd() + '/csv_FY/filter_bit/indicator_all_type_wrong.csv')
    criteria = ['has_data' + y for y in ['_13', '_14', '_15', '']]
    d = {'_13': 2013, '_14': '2014', '_15': '2015', '': '2013 - 2015'}
    for c in criteria:
        df_temp = df
        print 'before', len(df_temp)
        df_temp = df_temp[df_temp[c] == 1]
        print 'after', len(df_temp)
        df_temp.to_csv(os.getcwd() + '/csv_FY/filter_bit/{0}.csv'.format(c), index=False)
        df_temp = df_temp[['Building Number', 'Self-Selected Primary Function']]

        total = df_temp['Self-Selected Primary Function'].value_counts().sum()
        print total
        print df_temp['Self-Selected Primary Function'].value_counts()
        percent = df_temp['Self-Selected Primary Function'].value_counts()/total * 100
        d_percent = percent.to_dict()
        sorted_percent = sorted(d_percent.items(), key=operator.itemgetter(1))
        sorted_percent.reverse()
        labels = [x[0] for x in sorted_percent]
        fracs = [x[1] for x in sorted_percent]

        colors = sns.color_palette('Set3', len(labels))
        patches, texts = plt.pie(fracs, colors=colors, startangle=90)
        leg = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(labels, fracs)]

        plt.axis('equal')
        plt.legend(patches, leg, fontsize=8,bbox_to_anchor=(0.25, 1))
        sufname = c[len('has_data'):]
        plt.title('Pie chart of building type for {0}: {1} buildings'.format(d[sufname], total), x=0.5, y=1.05, fontsize=15)
        P.savefig(os.getcwd() + '/plot_FY_pie/buildingtype_{0}.png'.format(sufname), dpi = 75)
        plt.close()

def main():
    read_type()
    plot_type()
    return

main()
