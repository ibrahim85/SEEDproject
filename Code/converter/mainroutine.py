import os
import template2json as tm
import postProcess as ps
import PM2template as pm

# read PM excel to template excel
pm_in_dir = os.getcwd() + "/PMfile/PMinput/"
pm_out_dir = os.getcwd() + "/PMfile/PMoutput/"
pm.readPM(pm_in_dir, pm_out_dir)

# process template excel
json_in_dir = pm_out_dir
json_out_dir = os.getcwd() + "/Jsonfile/raw/"
tm.pm2json(json_in_dir, json_out_dir)

# post-process json files
post_out_dir = os.getcwd() + "/Jsonfile/post/"
ps.postProcess(json_out_dir, post_out_dir)
