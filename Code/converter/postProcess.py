import os
import converter as cv

dir_name = os.getcwd() + "/Jsonfile/"
in_file_name = cv.write_file_name
out_file_name = "post_test.txt"

# convert unit of time from 'ns' to 's'
def ns2s(string):
    return string.replace('000000000,', ',')

def postProcess():
    with open (dir_name + in_file_name, 'r+') as infile:
        unit_ns = infile.readline()
        unit_s = ns2s(unit_ns)
    with open (dir_name + out_file_name, 'w') as out:
        out.write(unit_s)
