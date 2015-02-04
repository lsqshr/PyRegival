from os.path import *
import os
from models import *
import csv, time
from regival import *
import time

dbpath = join('tests', 'testdata', '1P')

print '===FSL==='
reg = MrRegival(dbpath=dbpath)

if not os.path.exists(join(dbpath, 'ptemplate.pkl')):
	reg.build(normalise_method='FSL')
else:	
	reg.load_ptemplate(join(dbpath, 'ptemplate.pkl'))

reg.getcollection().write_meta()

pairs = reg.getcollection().find_transform_pairs(interval=[12])
targetpair = pairs[0]
reg.predict(targetpair, real_followupid='89591', option='change')

reg.printlog()

#os.remove(join(dbpath, 'ptemplate.pkl')) # Refresh
'''
print '\n===ANTS==='
reg = MrRegival(dbpath=dbpath)

if not os.path.exists(join(dbpath, 'ptemplate.pkl')):
	reg.build(normalise_method='ANTS')
else:	
	reg.load_ptemplate(join(dbpath, 'ptemplate.pkl'))

pairs = reg.getcollection().find_transform_pairs(interval=[12])
targetpair = pairs[0]
reg.predict(targetpair, real_followupid='89591', option='change')

reg.printlog()
'''