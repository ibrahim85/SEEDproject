import os
import pandas as pd
import numpy as np
import glob
import read_fy_withcat as rd
import matplotlib.pyplot as plt
import pylab as P

import seaborn as sns

def get_office():
    filename = os.getcwd() + '/csv/all_column/sheet-0-all_col.csv'
    df = pd.read_csv(filename)
    df = df[['Property Name', 'Self-Selected Primary Function']]
    df['Property Name'] = df['Property Name'].map(lambda x: x.partition('          ')[0][:8])
    df = df[df['Self-Selected Primary Function'] == 'Office']
    print len(df)
    return set(df['Property Name'].tolist())

def plot_box():
    filename = os.getcwd() + '/csv_FY/join/join_2015.csv'
    df = pd.read_csv(filename)
    df = df[df['eui'] >= 20]
    program = ['GP', 'LEED', 'first fuel', 'Shave Energy',
               'GSALink Option(26)', 'GSAlink I(55)', 'E4', 'ESPC',
               'Energy Star']
    length = len(program)

    for col in program:
        df.boxplot(column='eui', by = [col])
        plt.ylim((0, 140))
        P.savefig(os.getcwd() + '/plot_FY_annual/box_{0}.png'.format(col),
                  dpi = 150)

    f, axarr = plt.subplots(1, 2 * length, sharey=True, figsize=(20, 10))
    fs = 9
    for i in range(length):
        program_yes = df[df[program[i]] == 1]
        t = program_yes['eui'].tolist()
        axarr[2 * i].boxplot(np.asarray(t))
        axarr[2 * i].set_title((program[i] + '\nY'), fontsize=fs)

        program_no  = df[df[program[i]] == 0]
        t = program_no['eui'].tolist()
        axarr[2 * i + 1].boxplot(np.asarray(t))
        axarr[2 * i + 1].set_title(('N'), fontsize=fs)
        plt.ylim((0, 140))
    P.savefig(os.getcwd() + '/plot_FY_annual/box_all.png')

# box_plot by program
def plot_box_pro(theme, ylim, office):
    my_dpi = 300
    sns.set_style("white")
    sns.set_context("paper", font_scale=0.8)
    sns.mpl.rc("figure", figsize=(10,5.5))
    filename = os.getcwd() + '/csv_FY/join/join_2015.csv'
    df = pd.read_csv(filename)
    df['GSALink'] = df.apply(lambda row: 1 if row['GSALink Option(26)'] + row['GSAlink I(55)'] > 0 else 0, axis=1)
    df.drop(['GSAlink I(55)', 'GSALink Option(26)'], axis=1, inplace=True)
    office_set = get_office()
    if office:
        df = df[df['Building Number'].isin(office_set)]

    if theme == 'eui':
        df = df[df['eui_elec'] >= 12]
        df = df[df['eui_gas'] >= 3]
    if theme == 'eui_water':
        df = df[df['eui_water'] >= 5]
    if theme == 'eui_gas':
        df = df[df['eui_gas'] >= 3]
    if theme == 'eui_elec':
        df = df[df['eui_elec'] >= 12]
    # plot only programs
    dfs = []
    df_temp = df
    df_temp = df_temp[df_temp['Total Programs_v2'] > 0]
    totalnum = len(set(df_temp['Building Number'].tolist()))
    program = ['LEED', 'GP', 'first fuel', 'GSALink', 'E4', 'ESPC',
               'Shave Energy', 'Energy Star']
    colors = sns.husl_palette(7, l=.5, s=.9) + [(104.0/255,104.0/255,104.0/255)]
    color_dict = dict(zip(program, colors))

    sizes = []
    for col in program:
        df_yes = df[df[col] == 1]
        df_yes['program'] = col
        dfs.append(df_yes)
        sizes.append((col, len(df_yes)))
    df_all = pd.concat(dfs, ignore_index=True)
    df_plot = df_all
    size_dict = dict(sizes)
    print size_dict

    # sort by median
    gr = df_plot.groupby('program')
    medians = [(name, group[theme].median()) for name, group in gr]
    s_medians = sorted(medians, key=lambda x: x[1])
    order = [x for (x, _) in s_medians]
    order.remove('Energy Star')
    order.append('Energy Star')
    print theme
    print order

    ordered_color = [color_dict[x] for x in order]
    df_plot = df_all[['program', theme]]

    bx = sns.boxplot(x = 'program', y = theme, data = df_plot, fliersize=0, order = order, palette = sns.color_palette(ordered_color))
    st = sns.stripplot(x = 'program', y = theme, data = df_plot,
                       jitter=0.2, edgecolor='gray',
                       color = 'gray', size=0.3, alpha=0.5, order=order)
    xticklabels = ['(n={0})\n{1}+'.format(size_dict[pro], pro) for pro in order]
    bx.set(xticklabels=xticklabels)
    for tick in bx.xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
    for tick in bx.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    plt.ylim((0, ylim))
    plt.title('Total {0} Buildings'.format(totalnum), fontsize=15)
    plt.ylabel(theme.upper(), fontsize=12)
    bx.xaxis.set_label_coords(0.5, -0.08)
    plt.xlabel('Program', fontsize=12)
    plt.ylim((0, ylim))
    if office:
        P.savefig(os.getcwd() + '/plot_FY_annual/office/box_program_{0}.png'.format(theme), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi))
    else:
        P.savefig(os.getcwd() + '/plot_FY_annual/box_program_{0}.png'.format(theme), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi))
    plt.close()

def plot_box_vio(theme, ylim, office):
    sns.set_style("white")
    #sns.set_context("paper", font_scale=3)
    sns.set_context("paper", font_scale=0.8)
    sns.mpl.rc("figure", figsize=(10,5.5))
    colors_2 = sns.husl_palette(7, l=.8, s=.9) + [(192.0/255,192.0/255,192.0/255)]
    colors_1 = sns.husl_palette(7, l=.5, s=.9) + [(104.0/255,104.0/255,104.0/255)]
    colors = [[x, y] for (x, y) in zip(colors_1, colors_2)]
    colors = reduce(lambda x, y: x + y, colors)
    office_set = get_office()

    filename = os.getcwd() + '/csv_FY/join/join_2015.csv'
    df = pd.read_csv(filename)
    print ('input length', len(df))
    df['GSALink'] = df.apply(lambda row: 1 if row['GSALink Option(26)'] + row['GSAlink I(55)'] > 0 else 0, axis=1)
    df.drop(['GSAlink I(55)', 'GSALink Option(26)'], axis=1, inplace=True)
    if office:
        df = df[df['Building Number'].isin(office_set)]

    if theme == 'eui':
        df = df[df['eui_elec'] >= 12]
        df = df[df['eui_gas'] >= 3]
        print ('filter eui', len(df))
        '''
        p = 0.1
        print '\n {0} {1} percentile'.format(theme, p)
        print df[theme].quantile(p)
        '''
    if theme == 'eui_water':
        '''
        df = df[df['eui_water'] > 0]
        p = 0.1
        print '\n {0} {1} percentile'.format(theme, p)
        df = df[df['eui_water'] > 0]
        print df[theme].quantile(p)
        '''
        df = df[df[theme] >= 5]
        print ('filter water', len(df))
    if theme == 'eui_gas':
        df = df[df[theme] >= 3]
        print ('filter gas', len(df))
    if theme == 'eui_elec':
        df = df[df[theme] >= 12]
        print ('filter elec', len(df))

    totalnum = len(set(df['Building Number'].tolist()))
    print totalnum
    program = ['LEED', 'GP', 'first fuel', 'GSALink', 'E4', 'ESPC',
               'Shave Energy', 'Energy Star']
    ps = [[x + '+', ''] for x in program]
    ps = reduce(lambda x, y: x + y, ps)
    dfs = []
    sizes = []
    p_inc = []
    for col in program:
        df_yes = df[df[col] == 1]
        df_yes['program'] = col
        df_no = df[df[col] == 0]
        df_no['program'] = 'no_' + col
        dfs.append(df_yes)
        dfs.append(df_no)
        sizes.append(len(df_yes))
        sizes.append(len(df_no))
        percent_inprove = 0
        if df_no[theme].median() != 0:
            percent_inprove = (df_no[theme].median() - df_yes[theme].median())/df_no[theme].median()
        p_inc.append(percent_inprove)
    df_all = pd.concat(dfs, ignore_index=True)
    df_plot = df_all[['program', theme]]
    p_inc = [[str(round(x, 4)*100) + '%', ''] for x in p_inc]
    p_inc = reduce(lambda x, y:x + y, p_inc)
    yn = ['Yes', 'No'] * len(program)

    my_dpi = 300
    bx = sns.boxplot(x = 'program', y = theme, data = df_plot, fliersize=0, palette = sns.color_palette(colors))
    st = sns.stripplot(x = 'program', y = theme, data = df_plot,
                       jitter=0.2, edgecolor='gray',
                       color = 'gray', size=0.3, alpha=0.5)
    xticklabels = ['{0}(n={1})\n                 {2}\n                 {3}'.format(indi, size, p, p_i) for indi, size, p, p_i in zip(yn, sizes, ps, p_inc)]
    bx.set(xticklabels=xticklabels)
    for tick in bx.xaxis.get_major_ticks():
        tick.label.set_fontsize(7)
    for tick in bx.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    plt.title('Total {0} Buildings'.format(totalnum), fontsize=15)
    plt.ylabel(theme.upper(), fontsize=12)
    bx.xaxis.set_label_coords(0.5, -0.08)
    plt.xlabel('Program', fontsize=12)
    plt.ylim((0, ylim))

    if office:
        P.savefig(os.getcwd() + '/plot_FY_annual/office/box_all_{0}.png'.format(theme), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi))
    else:
        P.savefig(os.getcwd() + '/plot_FY_annual/box_all_{0}.png'.format(theme), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi))
    plt.close()

def plot_elec_eui_ratio():
    sns.set_palette(sns.color_palette('Blues', 3))
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)
    sns.mpl.rc("figure", figsize=(10,5))

    filelist = glob.glob(os.getcwd() + '/csv_FY/agg/*.csv')
    dfs = [pd.read_csv(csv) for csv in filelist]
    df_all = pd.concat(dfs, ignore_index=True)
    df_plot = df_all
    df_plot = df_plot[df_plot['eui'] < 10000]
    df_plot = df_plot[df_plot['eui'] >= 20]
    df_plot['ratio'] = df_plot['eui_elec']/df_plot['eui']
    #bx = sns.distplot(df_plot['ratio'], norm_hist=True)
    #df_plot.hist(column='ratio')
    for p in [0.25, 0.5, 0.75]:
        print (p, df_plot['ratio'].quantile(p))
    df_plot.boxplot(column='ratio')
    plt.show()
    plt.close()

# box plot by region, with different global cutoff for elec and gas
def plot_eui_region_filter_elecgas(col, ylim, office):
    sns.set_palette(sns.color_palette('Blues', 3))
    sns.set_style("white")
    sns.set_context("paper", font_scale=0.8)
    sns.mpl.rc("figure", figsize=(10,5))

    filelist = glob.glob(os.getcwd() + '/csv_FY/agg/*.csv')
    dfs = [pd.read_csv(csv) for csv in filelist]
    df_all = pd.concat(dfs, ignore_index=True)
    df_plot = df_all
    indicator = pd.read_csv(os.getcwd() + '/csv_FY/filter_bit/indicator_all.csv')
    df_bd = indicator[indicator['good_both'] == 1]
    if office:
        df_bd = df_bd[df_bd['office'] == 1]
    bd_set = set(df_bd['Building Number'].tolist())
    df_plot = df_plot[df_plot['Building Number'].isin(bd_set)]
    totalnum = len(set(df_plot['Building Number'].tolist()))
    st = sns.stripplot(x = 'Region No.', y = col, hue = 'Fiscal Year',
                       data = df_plot, jitter=0.2, edgecolor='gray',
                       color = 'gray', size=0.3, alpha=0.5)

    bx = sns.boxplot(x = 'Region No.', y = col, hue = 'Fiscal Year',
                     data = df_plot, fliersize=0)
    sizes = df_plot.groupby(['Region No.', 'Fiscal Year']).size()
    xticklabels = ['n={0}'.format(size) for group, size in sizes.iteritems()]
    xticklabels = ['{0}  {1}  {2}\nR{3}'.format(xticklabels[i], xticklabels[i + 1], xticklabels[i + 2], i + 1) for i in range(len(sizes)/3)]
    bx.set(xticklabels=xticklabels)
    plt.legend(loc = 2, bbox_to_anchor=(1, 1))
    if 'water' in col:
        plt.yticks([10 * x for x in range(4)])
    else:
        plt.yticks([20 * x for x in range(8)])
    for tick in bx.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)
    for tick in bx.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    plt.ylim((0, ylim))
    plt.title('{0} by region and year: total {1} buildings'.format(col, totalnum), fontsize=15)
    plt.ylabel(col.upper(), fontsize=12)
    my_dpi = 30
    if office:
        P.savefig(os.getcwd() + '/plot_FY_annual/office/office_{0}_region.png'.format(col), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi))
    else:
        P.savefig(os.getcwd() + '/plot_FY_annual/{0}_region.png'.format(col),
                  dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi))
    plt.close()

# box plot by region, with different cutoff for each energy type
def plot_eui_region(col, ylim, cut, strict):
    sns.set_palette(sns.color_palette('Blues', 3))
    sns.set_style("white")
    sns.set_context("paper", font_scale=0.7)
    sns.mpl.rc("figure", figsize=(10,5))

    filelist = glob.glob(os.getcwd() + '/csv_FY/agg/*.csv')
    dfs = [pd.read_csv(csv) for csv in filelist]
    df_all = pd.concat(dfs, ignore_index=True)
    df_plot = df_all
    df_plot = df_plot[df_plot['eui'] < 10000]
    if strict:
        df_plot = df_plot[df_plot[col] > cut]
    else:
        df_plot = df_plot[df_plot[col] >= cut]
    totalnum = len(set(df_plot['Building Number'].tolist()))
    st = sns.stripplot(x = 'Region No.', y = col, hue = 'Fiscal Year',
                       data = df_plot, jitter=0.2, edgecolor='gray',
                       color = 'gray', size=0.3, alpha=0.5)

    bx = sns.boxplot(x = 'Region No.', y = col, hue = 'Fiscal Year',
                     data = df_plot, fliersize=0)
    sizes = df_plot.groupby(['Region No.', 'Fiscal Year']).size()
    #xticklabels = ['{0}\nn={1}'.format(group, size) for group, size in sizes.iteritems()]
    xticklabels = ['n={0}'.format(size) for group, size in sizes.iteritems()]
    xticklabels = ['{0}  {1}  {2}\nR{3}'.format(xticklabels[i], xticklabels[i + 1], xticklabels[i + 2], i + 1) for i in range(len(sizes)/3)]
    bx.set(xticklabels=xticklabels)
    plt.legend(loc = 2, bbox_to_anchor=(1, 1))
    if 'water' in col:
        plt.yticks([10 * x for x in range(4)])
    else:
        plt.yticks([20 * x for x in range(8)])
    plt.ylim((0, ylim))
    plt.title('{0} by region and year: total {1} buildings'.format(col, totalnum))
    my_dpi = 300
    P.savefig(os.getcwd() + '/plot_FY_annual/{0}_region.png'.format(col),
              dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi))
    plt.close()

    '''
    g = sns.FacetGrid(df_plot, col='Region No.', size=4, aspect=0.4)
    g.map(sns.boxplot, 'Region No.', 'eui', 'Fiscal Year')
    g.map(sns.stripplot, 'Region No.', 'eui', 'Fiscal Year')
    plt.ylim((0, 140))
    plt.show()

    P.savefig(os.getcwd() + '/plot_FY_annual/eui_region_sub.png',
              dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi))
    plt.close()

    sns.violinplot(x = 'Region No.', y = 'eui', hue = 'Fiscal Year',
                   data = df_plot)
    plt.ylim((0, 140))
    my_dpi = 300
    P.savefig(os.getcwd() + '/plot_FY_annual/eui_region_vio.png',
              dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi))
    plt.close()
    '''
def region_boxplot():
    themes = ['eui', 'eui_elec', 'eui_gas', 'eui_water']
    ylims = [140, 120, 120, 30]
    for theme, ylim in zip(themes, ylims):
        plot_eui_region_filter_elecgas(theme, ylim, True)
        plot_eui_region_filter_elecgas(theme, ylim, False)

def main():
    #plot_box()
    themes = ['eui', 'eui_elec', 'eui_gas', 'eui_water']
    ylims = [140, 100, 100, 30]
    for theme, ylim in zip(themes, ylims):
        '''
        plot_box_vio(theme, ylim, False)
        plot_box_vio(theme, ylim, True)
        '''
        plot_box_pro(theme, ylim, False)
        plot_box_pro(theme, ylim, True)

    #region_boxplot()
    #plot_elec_eui_ratio()

main()
