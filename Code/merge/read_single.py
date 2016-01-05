import pandas as pd
import os
import glob
def read_static():
    indir = os.getcwd() + '/csv/select_column/'
    csv = indir + 'sheet-0.csv'
    print('read static info')
    df = pd.read_csv(csv)
    df['Property Name'] = df['Property Name'].map(lambda x: x.partition(' ')[0][:8])
    df['Postal Code'] = df['Postal Code'].map(lambda x: x[:5])

    df.rename(columns={'Property Name' : 'Building ID',
                       'State/Province' : 'State',
                       'Gross Floor Area' : 'GSF'}, inplace=True)

    regionfile = os.getcwd() + '/input/stateRegion.csv'
    print('reading region look up table')
    df_region = pd.read_csv(regionfile, usecols=['State', 'Region'])

    df_set = set(df['State'].tolist())
    region_set = set(df_region['State'].tolist())
    common_state_set = df_set.intersection(region_set)
    df = df[df['Country'] == 'United States']

    df = pd.merge(df, df_region, on='State')

    static_info_file = os.getcwd() + '/csv/cleaned/static_info.csv'
    df.to_csv(static_info_file, index=False)
    return df

def read_energy():
    energy_types = ['Electric - Grid', 'Natural Gas', 'Fuel Oil (No. 2)',
                    'Water']

    files = []
    for energy_type in energy_types:
        filelist = glob.glob(os.getcwd() + '/csv/single_building_sub/' + energy_type + '/' + '*.csv')
        frames = [pd.read_csv(f) for f in filelist]
        output = pd.concat(frames, ignore_index=True)
        files.append(output)
        energy_info_file = os.getcwd() + '/csv/cleaned/energy/' + energy_type[:4] + '.csv'
        output.to_csv(energy_info_file, index=False)

    result = pd.concat(files, ignore_index=True)
    result.to_csv(os.getcwd() + '/csv/cleaned/energy_info.csv', index=False)
    return result

def main():
    euas = os.getcwd() + '/input/EUAS.csv'
    print('Read in EUAS region')
    df_t = pd.read_csv(euas, usecols=['Building ID'])
    df_t['Building ID'] = df_t['Building ID'].map(lambda x: x.partition(' ')[0][:8])
    euas_set = set(df_t['Building ID'].tolist())

    df_static = read_static()
    df_energy = read_energy()
    df_merge = pd.merge(df_energy, df_static, on='Portfolio Manager ID')
    df_merge.info()
    df_base = df_merge.drop(['Usage/Quantity', 'Usage Units', 'Cost ($)',
                             'Portfolio Manager Meter ID', 'Meter Type'],
                            axis=1, inplace=False)
    df_base = df_base.drop_duplicates()
    df_base.info()

    grouped = df_merge.groupby('Meter Type')
    df_01 = grouped.get_group('Electric - Grid')
    df_01.rename(columns={'Usage/Quantity':'elec_amt',
                          'Usage Units':'elec_unit',
                          'Cost ($)':'elec_cost',
                          'Portfolio Manager Meter ID':'elec_meter_id'},
                 inplace=True)
    df_01.drop(['Building ID', 'State', 'Country', 'Postal Code',
                'Year Built', 'GSF', 'Region'],
               axis=1, inplace=True)
    df_01.info()
    merge_01 = pd.merge(df_base, df_01, how='left', on=['End Date',           'Portfolio Manager ID'])
    merge_01.info()

    df_02 = grouped.get_group('Natural Gas')
    df_02.rename(columns={'Usage/Quantity':'gas_amt',
                          'Usage Units':'gas_unit',
                          'Cost ($)':'gas_cost',
                          'Portfolio Manager Meter ID':'gas_meter_id'},
                 inplace=True)
    df_02.drop(['Building ID', 'State', 'Country', 'Postal Code',
                'Year Built', 'GSF', 'Region', 'Meter Type'],
               axis=1, inplace=True)
    merge_02 = pd.merge(merge_01, df_02, how='left', on=['End Date',          'Portfolio Manager ID'])
    merge_02.info()

    df_03 = grouped.get_group('Fuel Oil (No. 2)')
    df_03.rename(columns={'Usage/Quantity':'oil_amt',
                          'Usage Units':'oil_unit',
                          'Cost ($)':'oil_cost',
                          'Portfolio Manager Meter ID':'oil_meter_id'},
                 inplace=True)
    df_03.drop(['Building ID', 'State', 'Country', 'Postal Code',
                'Year Built', 'GSF', 'Region', 'Meter Type'],
               axis=1, inplace=True)
    merge_03 = pd.merge(merge_02, df_03, how='left', on=['End Date',          'Portfolio Manager ID'])
    merge_03.info()

    df_04 = grouped.get_group('Potable: Mixed Indoor/Outdoor')
    df_04.rename(columns={'Usage/Quantity':'water_amt',
                          'Usage Units':'water_unit',
                          'Cost ($)':'water_cost',
                          'Portfolio Manager Meter ID':'water_meter_id'},
                 inplace=True)
    df_04.drop(['Building ID', 'State', 'Country', 'Postal Code',
                'Year Built', 'GSF', 'Region', 'Meter Type'],
               axis=1, inplace=True)
    merge_04 = pd.merge(merge_03, df_04, how='left', on=['End Date',          'Portfolio Manager ID'])
    merge_04.drop(['Meter Type', 'Country'], axis=1, inplace=True)
    merge_04 = merge_04[merge_04['Building ID'].isin(euas_set)]
    print('Number of buildings after filter with EUAS membership')
    print(len(merge_04['Building ID'].unique()))
    merge_04.info()
    merge_04.to_csv(os.getcwd() + '/csv/testmerge-euas-2.csv', index=False)

main()
