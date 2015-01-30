from os.path import *
import os
from models import *
from preprocessing import *
import csv, time
from regival import *
from mrpredict import *
import time
import datetime

#dbpath = join('tests', "testdata", "5People")
#dbpath = join('tests', "testdata", "5ADNI-Patients")
dbpath = join('tests', "testdata", "4092cMCI-GRAPPA2")
#dbpath = join('tests', "testdata", "1Person-duplicated-images")

reg = MrRegival(dbpath=dbpath)
reg.starttime = datetime.datetime(2015,1,30, 15,0,0)

reg.autoredonorm()
'''
if not os.path.exists(join(dbpath, 'ptemplate.pkl')):
	reg.build(normalise_method='ANTS')
else:	
	reg.load_ptemplate(join(dbpath, 'ptemplate.pkl'))

pairs = reg.getcollection().find_transform_pairs(interval=[12])
targetpair = pairs[0]
reg.predict(targetpair, real_followupid='89591', option='change')

reg.printlog()

'''
