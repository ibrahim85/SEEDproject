import os
import glob

# convert unit of time from 'ns' to 's'
def ns2s(string):
    return string.replace('000000000,', ',')

def postProcess(in_dir, out_dir):
    filelist = glob.glob(in_dir + "*.txt")
    for file_in in filelist:
        with open (file_in, 'r+') as r_in:
            unit_ns = r_in.readline()
            unit_s = ns2s(unit_ns)
        file_out = file_in[len(in_dir):-4] + '_post.txt'
        with open (out_dir + file_out, 'w') as out:
            out.write(unit_s)
