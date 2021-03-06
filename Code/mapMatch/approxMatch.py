## ## ## ## ## ## ## ## ## ## ##
## ## Usage ## ##
## ## ## ## ## ## ## ## ## ## ##
##
## Run approxMatch with '-v' for verbose mode: python -v approxMatch
## Run approxMatch with '-d' for debug mode:python -d approxMatch
##
## Supported options:
##   -v, --verbose  show info
##   -d, --debug    debug printing
## ## ## ## ## ## ## ## ## ## ##

import os
import pandas as pd
from fuzzywuzzy import fuzz

## ## ## ## ## ## ## ## ## ## ##
## ## utilities ## ##
## ## ## ## ## ## ## ## ## ## ##

## ## ## ## ## ## ## ## ## ## ##
## logging and debugging print settings
import getopt
import logging
import sys

logger = logging.Logger('approxMatch')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Parse command line options.
opts, args = getopt.getopt(sys.argv[1:], 'vd', ['verbose', 'debug'])

# Map command line options to variables.
for option, argument in opts:
    if option in ('-v', '--verbose'):
        logger.setLevel(logging.INFO)
    elif option in ('-d', '--debug'):
        logger.setLevel(logging.DEBUG)
    else:
        assert False, "Unhandled option!"

def printDict(d, limit):
    count = 0
    iterator = iter(d)
    while count < limit:
        key = next(iterator)
        print '{0} -> {1}'.format(key, d[key])
        count += 1

## ## ## ## ## ## ## ## ## ## ##
## ## main ## ##
## ## ## ## ## ## ## ## ## ## ##
# read common address abbreviation table to dictionary
def readAddressLookup():
    logger.debug('\nreadAddressLookup')
    dirname = os.getcwd() + '/lookup/'
    # source of table http://pe.usps.gov/text/pub28/28apc_002.htm
    filename = 'addressAbbreviation.csv'
    df = pd.read_csv(dirname + filename)
    logger.debug("read address abbreviation:")
    logger.debug(df['Postal Service Standard Suffix Abbreviation'][:6])
    df2 = df.fillna(method = 'ffill') # forward filling missing data
    logger.debug(df2['Postal Service Standard Suffix Abbreviation'][:6])

    add_dict = dict(zip(df2['Commonly Used Street Suffix or Abbreviation'],
                        df2['Postal Service Standard Suffix Abbreviation']))
    return add_dict

def test_readAddressLookup(limit):
    print('\ntest_readAddressLookup')
    printDict(readAddressLookup(), limit)

add_abbrev_dict = readAddressLookup()

def preProcess(s, reason):
    if reason == 'address':
        # remove the trailing dot
        if s[len(s) - 1] == '.':
            s = s[:-1]
        L = s.split()
        logger.debug('tokenize: ')
        logger.debug('input: {0}, output: {1}'.format(s, L))
        acc = []
        for word in L:
            try:
                acc.append(add_abbrev_dict[word.upper()])
            except KeyError:
                logger.debug('{0} not an address abbreviation'.format(word.upper()))
                acc.append(word)
        return ' '.join(acc)
    else:
        return s

def test_preProcess():
    print('\ntest_preProcess')
    dirname = os.getcwd() + '/testInput/'
    # source of input: randomly picked points from Google Maps
    filename = 'address.csv'
    df = pd.read_csv(dirname + filename)
    print("read randomly picked address:")
    print(df['input'])
    df['output'] = df['input'].map(lambda x: preProcess(x, 'address'))
    print(df)
    return

# approximate string similarity evaluation with fuzzywuzzy package
def similarity(s, t, method):
    if method == 'partial':
        return fuzz.partial_ratio(s, t)
    elif method == 'token_sort':
        return fuzz.token_sort_ratio(s, t)
    elif method == 'token_set':
        return fuzz.token_set_ratio(s, t)
    else:
        return fuzz.ratio(s, t)

def test_similarity(s, t):
    print('\ntest_similarity({0}, {1})').format(s, t)
    s1 = preProcess(s, 'address')
    print 'original: {0}, after preProcess: {1}'.format(s, s1)
    methods = ['simple', 'partial', 'token_sort', 'token_set']
    for method in methods:
        print 'match ratio with {0} is {1}'.format(method,
                                                   similarity(s1, t, method))
    return

def main():
    s = '5000 Forbes Avenue'
    t = '5000 Forbes Ave'
    test_similarity(s, t)
    '''
    test_readAddressLookup(30)
    test_preProcess()
    '''

main()
