def get_range(df):
    print 'range for columns'
    for col in df:
        print '{0:>28} {1:>25} {2:>25}'.format(col, df[col].min(),
                                               df[col].max())
