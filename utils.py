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
        '''
        print('=====limgid==============')
        print(limgid) 
        print('===================')
        '''

        imgf = limg[ limgid.index(imgid) ]
        filepath = join(imgf[0], imgf[1])
        ladnimr.append(adnimrimg(meta, filepath))

    return ladnimr

def viewslice(file):
    path, fname = split(file)
    subprocess.Popen(["tkmedit",'-f', file])

# A collection tool class to perform some grouping and searching of the adnimriimg models
class AdniMrCollection(object):
    def __init__(self, lmodel):
        self.lmodel = lmodel 

    # Return Image.Data.ID pairs [[fixed_img_id1, moving_img_id1], [fixed_img_id2, moving_img_id2],...]
    def find_transform_pairs(self, interval=[6,12]):
        sbjdict = self.group_sbj()
        transpairs = []

        # Find the intervals with the specific lengths
        for key, sbj_imglist in sbjdict.iteritems():
            #sbj_imglist = sbjdict[key]
            sbj_imglist.sort(key=lambda x: x.getmetafield('VISCODE'))

            #print [ sbj[0] for sbj in sbj_imglist ] # Check the sorting

            for i in xrange(len(sbj_imglist)):
                for j in xrange(i+1,len(sbj_imglist)):
                    viscode1 = int(sbj_imglist[i].getmetafield('VISCODE').replace('m', ''))
                    viscode2 = int(sbj_imglist[j].getmetafield('VISCODE').replace('m', ''))
                    if (viscode2 - viscode1) in interval:
                        transpairs.append((sbj_imglist[j], sbj_imglist[i]))

        return transpairs

    # Group models into an RID dictionary <RID, [(VISCODE1, Image.Data.ID1), ...]> 
    def group_sbj(self):
        sbjdict = {}
        for model in self.lmodel:
            rid     = model.getmetafield('RID') 

            if rid in sbjdict:
                sbjdict[rid].append(model)
            else:
                sbjdict[rid] = [model]

        return sbjdict


