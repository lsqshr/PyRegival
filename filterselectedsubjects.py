from models import *
import os
from os.path import *
import shutil

# Load the selected subjects from csv
dbpath = '/media/siqi/SiqiLarge/ADNI-For-Pred'
c = AdniMrCollection(dbpath=dbpath, regendb=False)
mrlist = c.getmrlist()
lsbj = [m.getmetafield('Subject') for m in mrlist] # Extract the subject ID

# Filter the folders in dbpath
filtered_folders = [join(dbpath, fd) for fd in os.listdir(dbpath) if fd in lsbj]

# Copy the filtered folders to ../ADNI-For-Pred
for fd in filtered_folders:
    split(fd)[-1]
    shutil.copytree(fd, join(dbpath, '..', 'selected', split(fd)[-1]))

