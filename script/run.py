import os
import sys
argv = sys.argv
import datetime as dt
now = dt.datetime.now().strftime('%Y%m%d%H%M%S')

files = argv[1:]
command = 'nohup sh -c "'
for file in files:
    command += f'python -u {file} > log/{file}.{now}.log;'
command +='" &'
os.system(command)