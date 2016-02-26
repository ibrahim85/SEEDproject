import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import pylab as P
import seaborn as sns
import textwrap as tw

# FIX COLOR system
def read_ecm_header():
    return list(pd.read_csv(os.getcwd() + '/input/FY/ScopePortfolioReport_20160105-5.csv', header = [0, 1, 2], nrows=5))

# separate files into different level 1 sheets
def separate_files(names):
    df = pd.read_csv(os.getcwd() + '/csv_FY/join/join_ecm_keeplarge_2015.csv')
    level_1 = set([x[0] for x in names])
    for sheet in level_1:
        if (sheet == 'Building ID' or sheet == 'Project Type'):
            continue
        cols = [x for x in df if x[:x.find('_')] == sheet]
        all_cols = cols + ['eui_elec', 'eui_gas', 'eui', 'eui_water', 'Region No.', 'Fiscal Year', 'Building Number', 'Cat']
        df.to_csv(os.getcwd() + '/csv_FY/join/join_ecm_sep/{0}.csv'.format(sheet), index=False, cols=all_cols)

def get_level1(path):
    start = path.rfind('/') + 1
    end = path.find('.csv')
    return path[start:end]

def num_too_small(limit):
    df = pd.read_csv(os.getcwd() + '/csv_FY/join/join_ecm_2015.csv')
    df = df.ix[:, 23:]
    df.fillna(0, inplace=True)
    small_col = []
    for col in df:
        colsum = df[col].sum()
        if colsum < limit:
            print(col)
            small_col.append(col)
    return small_col

def remove_too_small(cols):
    df = pd.read_csv(os.getcwd() + '/csv_FY/join/join_ecm_2015.csv')
    df.drop(cols, axis=1, inplace=True)
    df.to_csv(os.getcwd() + '/csv_FY/join/join_ecm_keeplarge_2015.csv',
              index=False)

# re organize level2, action name, add a column of no-op for each action
def reorg_level2(names):
    filelist = glob.glob(os.getcwd() + '/csv_FY/join/join_ecm_sep/*.csv')
    for csv in filelist:
        df = pd.read_csv(csv)
        level1 = get_level1(csv)
        level2_list = list(set([x[1] for x in names if x[0] == level1]))
        #print '{0}:\n    {1}'.format(level1, level2_list)
        for level2 in level2_list:
            level3_list = list(set([x[2] for x in names \
                                   if x[0] == level1 and x[1] == level2]))
            #print '{0}-{1}:\n    {2}'.format(level1, level2, level3_list)
            non_col_name = '_'.join([level1, level2, 'No'])
            full_cols = ['_'.join([level1, level2, level3]) \
                         for level3 in level3_list]
            print full_cols
            for col in full_cols:
                df[col].fillna(0, inplace=True)
            df[non_col_name] = df.apply(lambda r: \
                    1 if sum([r[op] for op in full_cols]) == 0 else 0, axis=1)
            df.to_csv(os.getcwd() + \
                      '/csv_FY/join/join_ecm_reorg/{0}.csv'.format(level1),
                      index=False, encoding='utf-8')

ylim_dict = {'eui': 140, 'eui_elec': 140, 'eui_gas': 140, 'eui_water': 30}
def plot_dist(theme, office, plottype, limit):
    sns.set_style("white")
    sns.set_context("paper", font_scale=0.8)
    sns.mpl.rc("figure", figsize=(8,5))
    office_set = get_office()

    filelist = glob.glob(os.getcwd() + '/csv_FY/join/join_ecm_reorg/*.csv')
    for csv in filelist:
        df = pd.read_csv(csv)
        if office:
            df = df[df['Building Number'].isin(office_set)]
        if theme == 'eui':
            df = df[df['eui_elec'] >= 12]
            df = df[df['eui_gas'] >= 3]
            #print ('filter eui', len(df))
        if theme == 'eui_water':
            df = df[df[theme] >= 5]
            #print ('filter water', len(df))
        if theme == 'eui_gas':
            df = df[df[theme] >= 3]
            #print ('filter gas', len(df))
        if theme == 'eui_elec':
            df = df[df[theme] >= 12]
            #print ('filter elec', len(df))
        all_col_list = list(df)
        unchange_col_list = ['eui_elec', 'eui_gas', 'eui', 'eui_water',
                             'Region No.', 'Fiscal Year', 'Building Number',
                             'Cat']
        #print 'before drop cols'
        #print len(list(df))
        target_cols = list(set(all_col_list).difference(set(unchange_col_list)))
        too_small_cols = []
        def replace_suffix(colname):
            return colname[:colname.rfind('_')] + '_No'
        for col in target_cols:
            if ('_No' not in col) and (df[col].sum() < limit):
                too_small_cols.append(col)
        #print too_small_cols
        too_small_cols += [replace_suffix(x) for x in too_small_cols]
        df.drop(too_small_cols, axis=1, inplace=True)
        #print 'after drop cols'
        #print len(list(df))

        totalnum = len(set(df['Building Number'].tolist()))
        print totalnum
        print len(df)
        dfs = []
        p_inc = []
        emptys = []
        level1 = get_level1(csv)
        print '-----------------------------------'
        print level1
        level12 = list(set([header[:header.rfind('_')] for header in df \
                            if level1 in header]))
        program = level12
        all_program = []
        size_dict = {}
        median_dict = {}
        for pro in program:
            sizes = []
            sufs = []
            medians = []
            sub_program_list = [header for header in df \
                    if header[:header.rfind('_')] == pro]
            no_list = [x for x in sub_program_list if '_No' in x]
            if len(no_list) != 0:
                sub_program_list.remove(no_list[0])
                sub_program_list += no_list
            for sub in sub_program_list:
                df_yes = df[df[sub] == 1]
                if len(df_yes) == 0:
                    emptys.append(sum)
                    continue
                suf = sub[sub.rfind('_') + 1:]
                df_yes['sub'] = suf
                df_yes['program'] = pro
                dfs.append(df_yes)
                sufs.append(suf)
                sizes.append(len(df_yes))
                medians.append(df_yes[theme].median())
                #print (pro, suf, len(df_yes))
            sub_program_list = [x for x in sub_program_list if x not in emptys]
            size_dict[pro] = dict(zip(sufs, sizes))
            median_dict[pro] = dict(zip(sufs, medians))
            all_program += sub_program_list
        df_all = pd.concat(dfs, ignore_index=True)
        df_plot = df_all[['program', 'sub', theme]]
        sub_value = df_plot['sub'].unique()
        hue_order = sub_value.tolist()
        if 'No' in hue_order:
            hue_order.remove('No')
            hue_order.append('No')

        my_dpi = 300
        if plottype == 'box':
            bx = sns.boxplot(x = 'program', y = theme, hue = 'sub', hue_order = hue_order, data = df_plot, fliersize=0)
        elif plottype == 'vio':
            bx = sns.violinplot(x = 'program', y = theme, hue = 'sub', data = df_plot, fliersize=0)
        st = sns.stripplot(x = 'program', y = theme, hue = 'sub', hue_order = hue_order, data = df_plot, jitter=0.2, edgecolor='gray', color = 'gray', size=0.3, alpha=0.5)

        def median_reduce(median_d):
            assert('No' in median_d)
            no_median = median_d['No']
            del median_d['No']

            return dict([(key, str(round((no_median - median_d[key])/no_median * (-100), 2))) for key in median_d])

        # put No group to the end
        median_reduce_dict = {}
        size_str_dict = {}
        for k in size_dict:
            level2_dict = size_dict[k]
            flag = False
            if 'No' in level2_dict:
                no_size = level2_dict['No']
                del level2_dict['No']
                median_reduce_dict[k] = median_reduce(median_dict[k])
                flag = True
            else:
                keys = level2_dict.keys()
                median_reduce_dict[k] = dict(zip(keys, [''] * len(keys)))

            size_str_dict[k] = '\n'.join(['n({0})={1}, {2}%'.format(m, level2_dict[m], median_reduce_dict[k][m]) for m in level2_dict])
            if flag:
                size_str_dict[k] += '\nn({0})={1}'.format('No', no_size)

        if level1 == 'HVAC':
            print (theme, office)
            print 'median reduce'
            print median_reduce_dict
            print size_dict

        xticklabels = ['{0}\n{1}'.format('\n'.join(tw.wrap(p[p.rfind('_') + 1:], 20)), size_str_dict[p]) for p in program]
        bx.set(xticklabels=xticklabels)
        for tick in bx.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in bx.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        plt.title('{1}\nTotal {0} Buildings'.format(totalnum, level1), fontsize=15)
        plt.ylabel(theme.upper(), fontsize=12)
        bx.xaxis.set_label_coords(0.5, -0.09)
        plt.xlabel('', fontsize=10)
        plt.ylim((0, ylim_dict[theme]))
        plt.legend(loc = 2, bbox_to_anchor=(1, 1), fontsize=5)
        if office:
            P.savefig(os.getcwd() + '/plot_FY_annual/ECM2015_office/office_{1}_{0}_{2}plot.png'.format(theme, level1, plottype), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi))
        else:
            P.savefig(os.getcwd() + '/plot_FY_annual/ECM2015/{1}_{0}_{2}plot.png'.format(theme, level1, plottype), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi))
        plt.close()

def read_highlevel():
    df = pd.read_csv(os.getcwd() + \
                     '/input/FY/Portfolio HPGB Dashboard_highlevel.csv')
    df = df.ix[:, 6:]
    df.to_csv(os.getcwd() + \
              '/input/FY/Portfolio HPGB Dashboard_highlevel_select.csv',
              index=False)
    df = df[df['Advanced Metering'].notnull()]
    df.drop_duplicates(cols='Building ID').to_csv(os.getcwd() + \
              '/input/FY/Portfolio HPGB Dashboard_highlevel_dropdup.csv',
              index=False)

def read_ecm_highlevel():
    df_ecm = pd.read_csv(os.getcwd() + \
                         '/input/FY/Portfolio HPGB Dashboard_highlevel.csv')
    df_ecm = df_ecm.ix[:, 4:]
    df_ecm.drop('Project BA Code', axis=1, inplace=True)
    df_ecm = df_ecm[df_ecm['Advanced Metering'].notnull()]
    df_ecm.to_csv(os.getcwd() + \
            '/input/FY/Portfolio HPGB Dashboard_highlevel_dup_nona.csv',
            index=False)
    df_ecm.drop_duplicates(cols='Building ID', inplace=True)

    df_gsalink = pd.read_csv(os.getcwd() + \
            '/input/FY/GSAlink 81 Buildings Updated 9_22_15.csv')
    df_gsalink = df_gsalink[['Building ID']]
    df_gsalink['GSALink'] = 1
    df_ecm_gsa = pd.merge(df_ecm, df_gsalink, on = 'Building ID', how = 'outer')
    df_ecm_gsa['GSALink'].fillna(0, inplace=True)
    df_ecm_gsa.to_csv(os.getcwd() + \
            '/input/FY/Portfolio HPGB Dashboard_gsaLink.csv', index=False)

    ecm_cols = list(set(list(df_ecm)).difference(set(['Building ID',
                                                    'Facility ID'])))
    df_eui = pd.read_csv(os.getcwd() + '/csv_FY/join/join_2015.csv')
    df_all = pd.merge(df_eui, df_ecm_gsa, left_on = 'Building Number',
                      right_on='Building ID', how='left')
    for col in df_all:
        df_all[col].fillna(0, inplace=True)
    df_all.to_csv(os.getcwd() + '/csv_FY/join/join_ecm_2015_highlevel.csv', index=False)
    return

# each col is a program, in the returned df, there is a x_only column for each x in col, which indicates if the row has only 1 in x and 0 in all other cols

def get_col_only(df, all_cols, cols, outfile):
    df['total'] = df.apply(lambda row: reduce(lambda x, y: x + y, [row[x] for x in all_cols]), axis = 1)
    for c in cols:
        df[c + '_only'] = df.apply(lambda row: 1 if row[c] == 1 and
                                   row['total'] == 1 else 0, axis=1)
    df['None'] = df.apply(lambda row: 1 if row['total'] == 0 else 0,
                          axis=1)
    df.to_csv(outfile, index=False)

def getTwoECM():
    df = pd.read_csv(os.getcwd() + '/csv_FY/join/join_ecm_2015_highlevel.csv')
    outfile = os.getcwd() + '/csv_FY/join/join_2015_ecm_only.csv'
    # added GSALink
    all_cols = ['Advanced Metering', 'Building Envelope',
                'Building Tuneup or Utility Improvements', 'HVAC', 'IEQ',
                'Lighting', 'GSALink']
    cols = all_cols[:2] + all_cols[-1:]
    get_col_only(df, all_cols, cols, outfile)

#labels: a dictionary of labels of column
def plot_gsalink():
    import plot_dist_wtno as pdw
    themes = ['eui', 'eui_elec', 'eui_gas']
    ylims = [140, 140, 140]
    inputfile = os.getcwd() + '/csv_FY/join/join_2015_ecm_only.csv'

    '''
    columns = ['None', 'GSALink']
    labels = ['No ECM', 'GSALink']
    title = 'gsalink_noECM'
    for theme, ylim in zip(themes, ylims):
         pdw.plot_cols(inputfile, columns, labels, theme, title, ylim,
                       False, 'Blues', 4, 5.5)
         pdw.plot_cols(inputfile, columns, labels, theme, title, ylim,
                       True, 'Blues', 4, 5.5)
    '''

    columns = ['None', 'Advanced Metering_only']
    labels = ['No ECM', 'Advanced Metering_only']
    title = 'meter_noECM'
    colors = ['#f0bac2', '#c93d64']
    for theme, ylim in zip(themes, ylims):
         pdw.plot_cols(inputfile, columns, labels, theme, title, ylim,
                       False, sns.color_palette(colors), 4.5, 5.5)
         pdw.plot_cols(inputfile, columns, labels, theme, title, ylim,
                       True, sns.color_palette(colors), 4.5, 5.5)

    columns = ['None', 'Building Envelope_only']
    labels = ['No ECM', 'Building Envelope_only']
    title = 'enve_noECM'
    colors = ['#e1b258', '#886a32']
    for theme, ylim in zip(themes, ylims):
         pdw.plot_cols(inputfile, columns, labels, theme, title, ylim,
                       False, sns.color_palette(colors), 4.5, 5.5)
         pdw.plot_cols(inputfile, columns, labels, theme, title, ylim,
                       True, sns.color_palette(colors), 4.5, 5.5)

def plot_ecm_only():
    import plot_dist_wtno as pdw
    themes = ['eui', 'eui_elec', 'eui_gas']
    ylims = [140, 140, 140]
    program = ['Advanced Metering', 'Building Envelope', 'GSALink']

    inputfile = os.getcwd() + '/csv_FY/join/join_2015_ecm_only.csv'
    prefix = 'ecm_only'
    for theme, ylim in zip(themes, ylims):
         pdw.plot_box_vio(inputfile, program, prefix, theme, ylim, False)
         pdw.plot_box_vio(inputfile, program, prefix, theme, ylim, True)

def plot_ecm_highlevel(theme, ylim, office):
    sns.set_style("white")
    #sns.set_context("paper", font_scale=3)
    sns.set_context("paper", font_scale=0.8)
    sns.mpl.rc("figure", figsize=(10,5))
    colors_2 = sns.husl_palette(7, l=.8, s=.9) + \
            [(192.0/255,192.0/255,192.0/255)]
    colors_1 = sns.husl_palette(7, l=.5, s=.9) + \
            [(104.0/255,104.0/255,104.0/      255)]
    colors = [[x, y] for (x, y) in zip(colors_1, colors_2)]
    colors = reduce(lambda x, y: x + y, colors)

    office_set = get_office()

    df = pd.read_csv(os.getcwd() + '/csv_FY/join/join_ecm_2015_highlevel.csv')
    if office:
        df = df[df['Building Number'].isin(office_set)]
    if theme == 'eui':
        df = df[df['eui_elec'] >= 12]
        df = df[df['eui_gas'] >= 3]
        #print ('filter eui', len(df))
    if theme == 'eui_water':
        df = df[df[theme] >= 5]
        #print ('filter water', len(df))
    if theme == 'eui_gas':
        df = df[df[theme] >= 3]
        #print ('filter gas', len(df))
    if theme == 'eui_elec':
        df = df[df[theme] >= 12]
        #print ('filter elec', len(df))
    program = list(df)[-8:]
    program.remove('Renewable Energy')
    program.remove('Water')
    print program
    totalnum = len(set(df['Building Number'].tolist()))
    print totalnum

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
            percent_inprove = (df_no[theme].median() - df_yes[theme].median())/    df_no[theme].median()
        p_inc.append(percent_inprove)
    df_all = pd.concat(dfs, ignore_index=True)
    df_plot = df_all[['program', theme]]
    p_inc = [[str(round(x, 4)*100) + '%', ''] for x in p_inc]
    p_inc = reduce(lambda x, y:x + y, p_inc)
    p_inc_order = sorted(zip(program, p_inc), key=lambda x: x[1])
    yn = ['Yes', 'No'] * len(program)
    order = [x[0] for x in p_inc_order]

    my_dpi = 300
    bx = sns.boxplot(x = 'program', y = theme, data = df_plot, fliersize=0,palette = sns.color_palette(colors))
    #BOOKMARK
    st = sns.stripplot(x = 'program', y = theme, data = df_plot,
                       jitter=0.2, edgecolor='gray',
                       color = 'gray', size=0.3, alpha=0.5)
    xticklabels = ['{0}(n={1})\n                 {2}\n                 {3}'.format(indi, size, '\n'.join(tw.wrap(p, 30,subsequent_indent = '            ')), p_i) for indi, size, p, p_i in zip(yn, sizes, ps, p_inc)]
    bx.set(xticklabels=xticklabels)
    for tick in bx.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)
    for tick in bx.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    plt.title('Total {0} Buildings'.format(totalnum), fontsize=15)
    plt.ylabel(theme.upper(), fontsize=12)
    bx.xaxis.set_label_coords(0.5, -0.08)
    plt.xlabel('', fontsize=12)
    plt.ylim((0, ylim))

    if office:
        P.savefig(os.getcwd() + '/plot_FY_annual/ECM2015_office/office_highlevel_{0}_boxplot.png'.format(theme), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi))
        P.savefig(os.getcwd() + '/plot_FY_annual/ECM2015_office/office_highlevel_{0}_boxplot_droplast.png'.format(theme), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi))
    else:
        P.savefig(os.getcwd() + '/plot_FY_annual/ECM2015/highlevel_{0}_boxplot_droplast.png'.format(theme), dpi = my_dpi, figsize = (2000/my_dpi, 500/my_dpi))
    plt.close()

def read_ecm_high():
    names = read_ecm_header()
    names_concat = ['_'.join(x) for x in names]
    df_ecm = pd.read_csv(os.getcwd() + '/input/FY/ScopePortfolioReport_20160105-5.csv',
                         header = None, skiprows=3, names=names_concat)
    df_eui = pd.read_csv(os.getcwd() + '/csv_FY/join/join_2015.csv')
    df_all = pd.merge(df_eui, df_ecm, left_on='Building Number',
                      right_on=names_concat[0], how='left')
    df_all.to_csv(os.getcwd() + '/csv_FY/join/join_ecm_2015.csv', index=False)
    return

def get_office():
    filename = os.getcwd() + '/csv/all_column/sheet-0-all_col.csv'
    df = pd.read_csv(filename)
    df = df[['Property Name', 'Self-Selected Primary Function']]
    df['Property Name'] = df['Property Name'].map(lambda x: x.partition('          ')[0][:8])
    df = df[df['Self-Selected Primary Function'] == 'Office']
    print len(df)
    return set(df['Property Name'].tolist())

def colname2tuple(colname):
    idx_under_front = colname.find('_')
    idx_under_back = colname.rfind('_')
    return (colname[:idx_under_front],
            colname[idx_under_front + 1: idx_under_back],
            colname[idx_under_back + 1:])

def plot_ecm_lowlevel():
    names = read_ecm_header()
    too_small_list = num_too_small(1)
    too_small_list = [colname2tuple(x) for x in too_small_list]
    # cols with too small sum removed
    names = list(set(names).difference(set(too_small_list)))
    # return a list of col with count < 1
    #remove_too_small(num_too_small(1))
    #separate_files(names)
    #reorg_level2(names)
    #themes = ['eui']
    themes = ['eui', 'eui_elec', 'eui_gas', 'eui_water']
    for theme in themes:
        plot_dist(theme, False, 'box', 5)
        plot_dist(theme, True, 'box', 5)

def check_dup():
    df = pd.read_csv(os.getcwd() + '/input/FY/Portfolio HPGB Dashboard_highlevel_select.csv')
    df[df.duplicated(cols='Building ID') | df.duplicated(cols='Building ID', take_last=True)].to_csv(os.getcwd() + '/input/FY/Portfolio HPGB Dashboard_highlevel_dup.csv', index=False)
    df2 = pd.read_csv(os.getcwd() + '/input/FY/Portfolio HPGB Dashboard_highlevel_dup.csv')
    df2 = df2[df2['Advanced Metering'].notnull()]
    df2[df2.duplicated(cols='Building ID') | df2.duplicated(cols='Building ID', take_last=True)].to_csv(os.getcwd() + '/input/FY/Portfolio HPGB Dashboard_highlevel_dup_nona.csv', index=False)

def drop_na_completiondate():
    df = pd.read_csv(os.getcwd() + '/input/FY/completionDate.csv')
    df.dropna(inplace=True)
    df.to_csv(os.getcwd() + '/input/FY/completionDate_dropna.csv', index=False)
    df.drop_duplicates(cols='Building ID').to_csv(os.getcwd() + '/input/FY/completionDate_dropdup.csv', index=False)

def main():
    #plot_ecm_lowlevel()
    #themes = ['eui']

    '''
    themes = ['eui', 'eui_elec', 'eui_gas', 'eui_water']
    for theme in themes:
        plot_ecm_highlevel(theme, ylim_dict[theme], False)
        plot_ecm_highlevel(theme, ylim_dict[theme], True)
    #check_dup()
    #read_ecm_highlevel()
    '''
    #drop_na_completiondate()

    #read_ecm_highlevel()
    #getTwoECM()

    # plot_ecm_only()
    plot_gsalink()
    return

main()
