from os.path import *
import os
from models import *
import csv, time
from regival import *
import time
import itertools

NTEST = 1
interval = [12]
dbpath = '/media/siqi/SiqiLarge/ADNI-For-Pred'
ncore = 4

c = AdniMrCollection(dbpath=dbpath, regendb=False)
c.randomselect(60, interval)

reg = MrRegival(collection=c, dbpath=dbpath)
reg.normalise(ignoreexception=True, ncore=ncore)
reg.transform(ignoreexception=True, ncore=ncore)

epairs = reg.getcollection().filter_elligible_pairs(interval=interval)
testset = epairs[0:NTEST] # Try one case
templateset = epairs[NTEST:]

diffs = []
if NTEST == 1:
	diffs = list(itertools.product(testset, templateset))
else:
	for p in testset:
		diffs += list(itertools.product(p, templateset))

transdistance = reg.transdiff(diffs, option='trans', ignoreexception=False, ncore=4)
imagedistance = reg.transdiff(diffs, option='image', ignoreexception=False, ncore=4)

        
## Save the diffs
session = {}
session['testset'] = testset
session['templateset'] = templateset
session['g'] = g
session['transdistance'] = transdistance
session['imagedistance'] = imagedistance
session['reg'] = reg

with open(join(self.dbpath, 'expttemplate.pkl'), 'wb') as outfile:
    pickle.dump(session, outfile)


# Make the prediction
if NTEST == 1:
	testset = [testset]

dtemplate = zip(templateset, distance]

for p in testset:
	followup = c.find_followups([p], interval=interval)
	followid = followup.fixed_image.get_imgid()
	targettemplates = [t for t in dtemplateset if t[0][0] == p]
	tset = [t[0] for t in targettemplates]
	w = [t[1] for t in targettemplates]
	predictionerr = reg.predict(testset, tset, w, real_followupid=followid)
	print 'prediction err is', predictionerr