import pandas as pd
import os

indir = os.getcwd() + '/csv/single_building/'
infile = 'pm-1539.csv'
df = pd.read_csv(indir + infile)
grouped = df.groupby('Meter Type')

print 'data frame basic info'
df.info()
print 'group summary'
print grouped.size()

for name,group in grouped:
    print name
    print 'range of time: {0}, {1}'.format(group['End Date'].min(),
                                           group['End Date'].max())
