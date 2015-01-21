from os.path import *
from utils import *
from preprocessing import *
import csv, time
from regival import *
from mrpredict import *
import time

dbpath = join('tests', "testdata", "5People")
reg = MrRegival(dbpath=dbpath)
#ladnimr = reg.getcollection().getmrlist()

if not os.path.exists(join(dbpath, 'ptemplate.pkl')):
	reg = MrRegival(dbpath)
	reg.build(normalise_method='FSL')
else:	
	reg.load_ptemplate(join(dbpath, 'ptemplate.pkl'))

pairs = reg.getcollection().find_transform_pairs(interval=[12])
targetpair = pairs[0]
reg.predict(targetpair, real_followup='204131', option='change')