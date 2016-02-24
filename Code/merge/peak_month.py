import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import pylab as P
import seaborn as sns

sns.set_palette(sns.color_palette('Set2', 3))

# false_bd_set is a set of building with questionable eui or water
def get_peak(year, criteria, hilow):
    if criteria == 'eui':
        files = glob.glob(os.getcwd() + '/csv_FY/false_eui/false_eui_{0}.csv'.format(year))
        false_bd_set_list = [set((pd.read_csv(csv))['Building Number'].tolist()) for csv in files]
        false_bd_set = reduce(set.union, false_bd_set_list)
    elif criteria == 'all':
        files = glob.glob(os.getcwd() + '/csv_FY/false_eui/*_{0}.csv'.format(year))
        false_bd_set_list = [set((pd.read_csv(csv))['Building Number'].tolist()) for csv in files]
        false_bd_set = reduce(set.union, false_bd_set_list)
    else:
        false_bd_set = set([])
    print criteria, len(false_bd_set)

    filelist = glob.glob(os.getcwd() + '/csv_FY/single_eui_cal/*_{0}.csv'.format(year))
    p_eui = []
    p_elec = []
    p_gas = []
    p_oil = []
    p_water = []
    bd = []
    for csv in filelist:
        df = pd.read_csv(csv)
        # filter out false buildings
        filename = csv[csv.find('single_eui_cal') + len('single_eui_cal') + 1:]
        building = filename[:8]
        if hilow = 'high':
            peak_eui_month = df.ix[df['eui'].argmax(), 'month']
            peak_elec_month = df.ix[df['eui_elec'].argmax(), 'month']
            peak_gas_month = df.ix[df['eui_gas'].argmax(), 'month']
            peak_oil_month = df.ix[df['eui_oil'].argmax(), 'month']
            peak_water_month = df.ix[df['eui_water'].argmax(), 'month']
        else:
            peak_eui_month = df.ix[df['eui'].argmin(), 'month']
            peak_elec_month = df.ix[df['eui_elec'].argmin(), 'month']
            peak_gas_month = df.ix[df['eui_gas'].argmin(), 'month']
            peak_oil_month = df.ix[df['eui_oil'].argmin(), 'month']
            peak_water_month = df.ix[df['eui_water'].argmin(), 'month']
        #print (peak_eui_month, peak_elec_month, peak_gas_month,
        #       peak_oil_month, peak_water_month)
        p_eui.append(peak_eui_month)
        p_elec.append(peak_elec_month)
        p_gas.append(peak_gas_month)
        p_oil.append(peak_oil_month)
        p_water.append(peak_water_month)
        bd.append(building)

    df_peak = pd.DataFrame({'peak_eui_month': p_eui,
                            'peak_elec_month': p_elec,
                            'peak_gas_month': p_gas,
                            'peak_oil_month': p_oil,
                            'peak_water_month': p_water,
                            'building': bd})
    df_peak['year'] = year
    #print df_peak.head()
    df_peak['bad'] = df_peak['building'].map(lambda x: 1 if x in false_bd_set else 0)
    df_peak = df_peak[df_peak['bad'] == 0]
    df_peak.info()
    #df_peak.drop('bad', inplace=True)
    df_peak.to_csv(os.getcwd() + '/csv_FY/info_month/filter_{0}/peak_{1}.csv'.format(criteria, year), index=False)

def write_peak(criteria):
    for year in [2012, 2013, 2014, 2015]:
        get_peak(year, criteria)

def plot_peak(criteria):
    category = ['peak_eui_month', 'peak_elec_month', 'peak_gas_month',
                'peak_oil_month', 'peak_water_month']
    filelist = glob.glob(os.getcwd() + '/csv_FY/info_month/filter_{0}/*.csv'.format(criteria))
    dfs = [pd.read_csv(csv) for csv in filelist]
    df_all = pd.concat(dfs, ignore_index=True)
    '''
    for col in category:
        sns.violinplot(x = 'year', y = col, data = df_all)
        P.savefig(os.getcwd() + '/plot_FY/peak_month/violin/{0}.png'.format(col))
        plt.close()

        sns.boxplot(x = 'year', y = col, data = df_all)
        P.savefig(os.getcwd() + '/plot_FY/peak_month/box/{0}.png'.format(col))
        plt.close()
    '''
    for col in category:
        for df in dfs:
            yr = df.ix[0, 'year']
            sns.distplot(df[col])
            plt.title('{0}_{1} distribution'.format(col, yr))
            P.savefig(os.getcwd() + '/plot_FY/peak_month/dist/filter_{0}/{1}_{2}.png'.format(criteria, col, yr))
            plt.close()

def main():
    # write to peak file
    for criteria in ['eui', 'all', 'none']:
        write_peak(criteria)
        plot_peak(criteria)
    return

main()
