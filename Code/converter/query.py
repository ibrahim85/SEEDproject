import psycopg2
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
    # start connection
    try:
        # return a connection instance
        conn = psycopg2.connect("dbname='seed-deploy' user='seed-admin' host='127.0.0.1' password='SEEDDB@architecture.cmu.edu'")
    except psycopg2.Error as e:
        print "Unable to connect to database"
        print e.pgerror
        print e.diag.message_detail
        return

    for address in addresses:
        # open a cursor
        cur = conn.cursor()

        # retrieve GreenButton URL and building_id relationship from PostgreSQL
        # Query the database and obtain data as Python objects
        # select the table seed_buildingsnapshot and the record with
        # address equals to the query address
        # ?? is the table right ??
        # ?? is the column right ??
        # ?? is there a field called 'active'??
        # ?? one or two cursor to search for two tables ??
        cur.execute('select * from seed_buildingsnapshot where Address Line 1 = "{0}" and active="Y"'.format(address))

        # get buildingsnapshot_id
        try:
            # fetch one row of a query result
            row = cur.fetone()
            r1 = reg(cur_des, row)
            s_id = r1.id #snapshot_id
        except 'ProgrammingError':
            print 'No matching data with address {0}'.format(address)
            s_id = None
            c_id = None
            d[address] = {'buildingsnapshot_id' : s_id,
                          'canonical_building' : c_id}
            continue

        # query to get canonical id
        cur.execute('select * from seed_canonicalbuilding where canonical_snapshot_id = {0}'.format(buildingsnapshot_id))
        try:
            # fetch one row of a query result
            row = cur.fetone()
            r2 = reg(cur_des, row)
            c_id = r2.id #canonical_id
        except 'ProgrammingError':
            print 'No matching data with address {0}'.format(address)
            c_id = None
        d[address] = {'buildingsnapshot_id' : s_id,
                      'canonical_building' : c_id}

    cur.close()
    conn.close()
    print 'connection closed'

## ## ## ## ## ## ## ## ## ## ##
## helper and testing ##
## ## ## ## ## ## ## ## ## ## ##
import pandas as pd
import numpy as np
import os

# read in the sample csv table and return a dictionary
def getDictfromCSV():
    dirname = os.getcwd() + '/PMfile/'
    filename = 'covered-buildings-sample.csv'
    df = pd.read_csv(dirname + filename)
    ar = df['Address'].values
    d = {}
    for a in np.nditer(ar, flags=['refs_ok']):
        print a
        d['{0}'.format(a)] = None
    return d

def test_retrieveID():
    d = getDictfromCSV()
    # un-comment when the retrieveID function is varified
    # retrieveID()
    print d
    return

def main():
    #getDictfromCSV()
    #test_retrieveID()
    return 0

main()
