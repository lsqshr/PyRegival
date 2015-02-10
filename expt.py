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
#c.randomselect(60, interval)

## Template Construction
reg = MrRegival(collection=c, dbpath=dbpath)
#reg.normalise(ignoreexception=True, ncore=ncore)
'''
# Without cleaning transformed subjects would have more than one transid
import shutil
        if exists(join(self.dbpath, 'results')):
            shutil.rmtree(join(self.dbpath, 'results'))
'''
#reg.transform(ignoreexception=True, ncore=ncore)

epairs = reg.getcollection().filter_elligible_pairs(interval=interval)
testset = epairs[0:NTEST] # Try one case
templateset = epairs[NTEST:]

# itertools will return [(test1, template1), (test1, template2), (test1, template3) ... (testn, templaten)]
diffs = list(itertools.product(testset, templateset)) 

transdistance = reg.transdiff(diffs, option='trans', ignoreexception=False, ncore=ncore)
imagedistance = reg.transdiff(diffs, option='image', ignoreexception=False, ncore=ncore)

session = {}
session['testset'] = testset
session['templateset'] = templateset
session['transdistance'] = transdistance
session['imagedistance'] = imagedistance

with open(join(dbpath, 'expttemplate.pkl'), 'wb') as outfile:
    pickle.dump(session, outfile)


## Prediction
with open('expttemplate.pkl', 'rb') as infile:
    session = pickle.load(infile)

testset = session['testset']
templateset = session['templateset']
transdistance = session['transdistance']
imagedistance = session['imagedistance']

#dtemplateset = zip(templateset, transdistance)
distances = transdistance # Will be tuned

for i, p in enumerate(testset):
	followup = c.find_followups([p], interval=interval)
	followid = followup[0].fixedimage.getimgid()
	#targettemplates = [t for t in dtemplateset if t[0][0] == p]
	#tset = [t[0] for t in targettemplates]
    targettemplateset = templateset
	w = distances[i*len(templateset):(i+1)*len(templateset)]
	predictionerr = reg.predict(p, templateset, w, real_followupid=followid, ncore=ncore)
	print 'prediction err is', predictionerr
