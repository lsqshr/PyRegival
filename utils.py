import os
import subprocess
import re
from os.path import * 
import csv
from models import adnimrimg

# Traverse the dbpath for the files with provided suffix
def traverse_for_file(path, suffix):
    limg = [];
    for root, dirs, files in os.walk(path):
      for file in files:
        if file.endswith(suffix):
            limg.append((root, file))
    return limg


# Find Image ID from ADNI nii filename
def find_adni_imgid(fname):
    lmatch = re.findall('_I\d+', fname)
    assert len(lmatch) <= 1 , 'More than one matches were found: '
    cleanmatch = [] 

    for m in lmatch: 
      cleanmatch.append(m[2:]) # Remove the '_' before image id 

    return cleanmatch[0] if len(cleanmatch) == 1 else None


def get_adni_mrlist(dbpath):
    # Find all images stated in the dbgen.csv in the db path
    # Construct adnimrimg list
    limg = traverse_for_file(dbpath, '.nii')
    limgid = [find_adni_imgid(img[1]) for img in limg ]
    ladnimr = []

    # read in the dbgen.db
    with open(join(dbpath, 'dbgen.csv')) as f:
      r = csv.reader(f, delimiter=',')
      header = next(r)
      #imgididx = header.index('Image.Data.ID')

      for row in r:
        meta = {}
        for i, col in enumerate(row):
            meta[header[i]] = col

        imgid = meta['Image.Data.ID']

        print('=====limgid==============')
        print(limgid) 
        print('===================')

        imgf = limg[ limgid.index(imgid) ]
        filepath = join(imgf[0], imgf[1])
        ladnimr.append(adnimrimg(meta, filepath))
    return ladnimr

def viewslice(file):
    path, fname = split(file)
    subprocess.Popen(["tkmedit",'-f', file])

