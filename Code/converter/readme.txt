This is a brief document about the PM excel parser
author: yujie xu (yujiex@andrew.cmu.edu)

# file structure #
.py files:
mainroutine.py:
    root routine
PM2template.py:
    converting Energy PM file to template excel file
template2json.py:
    converting template excel file to json format
query.py:
    make query to seed database and retrieve buildingsnapshot_id
postProcess.py:
    post process json file,
    convert json time unit from 'ns' to 's'

input/output folders
/PMfile/PMinput/:
    hold EnergyStar PM files
    user should put the downloaded EnergyStar PM files in this folder
/PMfile/PMoutput/:
    hold EnergyStar PM data in excel template format
    this folder will initially be empty
/PMfile/template10field.xlsx:
    the newest template
/PMfile/covered-buildings-sample.csv:
    seed sample data, used for testing query.py
/PMfile/portfolio-manager-sample.csv:
    seed sample data, not used

/Jsonfile/raw/:
    hold raw output json file converted from excel template
/Jsonfile/post/:
    hold final output json file with the corrected time units

/archive/: for older versions of code and data files, not used

# how to run it #
type 'python mainroutine.py' in command line
