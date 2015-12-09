import psycopg2
import logging
## ## ## ## ## ## ## ## ## ## ##
## create logger for print ##
## ## ## ## ## ## ## ## ## ## ##
logger = logging.Logger('mainroutine')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

## ## ## ## ## ## ## ## ## ## ##
## main function ##
## ## ## ## ## ## ## ## ## ## ##
# from paser/task.py
class reg(object):
    def __init__(self, cursor_des, registro):
        for (attr, val) in zip((d[0] for d in cursor_des), registro):
            setattr(self, attr, val)

# from paser/task.py
# d: dictionary with address as its key and null as its initial value
# for each key in d, retrieve the buildingsnapshot_id and canonical id
def retrieveID(d):
    logger.info('\nretrieveID of buildings')
    # start connection
    try:
        # return a connection instance
        # host ip: 128.2.110.179
        conn = psycopg2.connect("dbname='seed-deploy' user='seed-admin' host='128.2.110.179' port='5432' password='SEEDDB@architecture.cmu.edu'")
        logger.info('connected to host: 128.2.110.179')
    except psycopg2.Error as e:
        print "Unable to connect to database"
        print e.pgerror
        print e.diag.message_detail
        return

    for address in d:
        # open a cursor
        cur = conn.cursor()

        # retrieve GreenButton URL and building_id relationship from PostgreSQL
        # Query the database and obtain data as Python objects
        # select the table seed_buildingsnapshot and the record with
        # address equals to the query address
        # use pgAdmin to check the content of the table in postgres
        # address search is a simple query, might need to change for
        # approximate searching later
        query_string = 'select * from seed_buildingsnapshot \
                where address_line_2 = \'{0}\''.format(address)
        logger.debug(query_string)
        cur.execute(query_string)

        # get buildingsnapshot_id
        try:
            # fetch one row of a query result
            row = cur.fetchone()
            cur_des = cur.description
            if row is not None:
                r1 = reg(cur_des, row)
                s_id = r1.id #snapshot_id
            cur.close()
        except 'ProgrammingError':
            print 'No matching data with address {0}'.format(address)
            s_id = None
            c_id = None
            d[address] = {'buildingsnapshot_id' : s_id,
                          'canonical_building' : c_id}
            cur.close()
            continue

        cur2 = conn.cursor()
        # query to get canonical id
        cur2.execute('select * from seed_canonicalbuilding where canonical_snapshot_id = {0}'.format(s_id))
        try:
            # fetch one row of a query result
            row = cur2.fetchone()
            if row is not None:
                cur_des = cur2.description
                r2 = reg(cur_des, row)
                c_id = r2.id #canonical_id
            else:
                c_id = None
        except 'ProgrammingError':
            print 'No matching data with address {0}'.format(address)
            c_id = None
        d[address] = {'buildingsnapshot_id' : s_id,
                      'canonical_building' : c_id}

        cur2.close()
        logger.debug((address, s_id, c_id))
    conn.close()
    logger.info('connection closed')

## ## ## ## ## ## ## ## ## ## ##
## helper and testing ##
## ## ## ## ## ## ## ## ## ## ##
import pandas as pd
import numpy as np
import os

# read in the sample csv table and return a dictionary
def getDictfromCSV():
    logger.debug('\ngetDictfromCSV:')
    dirname = os.getcwd() + '/PMfile/'
    filename = 'covered-buildings-sample.csv'
    df = pd.read_csv(dirname + filename)
    ar = df['Address'].values
    d = {}
    for a in np.nditer(ar, flags=['refs_ok']):
        logger.debug(a)
        d['{0}'.format(a)] = None
    return d

def test_retrieveID():
    logger.debug('\ntest_retrieveID:')
    d = getDictfromCSV()
    # un-comment when the retrieveID function is varified
    # print d
    retrieveID(d)
    return

def main():
    #getDictfromCSV()
    #test_retrieveID()
    return 0

main()
