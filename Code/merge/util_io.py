# print dictionary d with 'limit' number of records
def print_dict(d, limit):
    count = 0
    iterator = iter(d)
    while count < limit:
        key = next(iterator)
        print '{0} -> {1}'.format(key, d[key])
        count += 1
