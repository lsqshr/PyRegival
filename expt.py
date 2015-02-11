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
K = 2
TRIALSTART = 0
TRIALEND = 60

c = AdniMrCollection(dbpath=dbpath, regendb=False)
#c.randomselect(60, interval)

## Template Construction
reg = MrRegival(collection=c, dbpath=dbpath)
#reg.normalise(ignoreexception=True, ncore=ncore)
# Without cleaning transformed subjects would have more than one transid
'''
import shutil
if exists(join(self.dbpath, 'results', 'transforms')):
            shutil.rmtree(join(self.dbpath, 'results'))
'''
#reg.transform(ignoreexception=True, ncore=ncore)

epairs = reg.getcollection().filter_elligible_pairs(interval=interval)

# Collect Leave one out diff
for i, testpair in enumerate(epairs):
    if i < TRIALSTART:
        continue
    if i > TRIALEND:
        break;
    # session will be save after every leave-one-out trial
    if exists(join(dbpath, 'expttemplate.pkl')):
        with open(join(dbpath, 'expttemplate.pkl'), 'rb') as infile:
            session = pickle.load(infile)
    else:
        session = [0]*len(epairs)
    session[i] = {}
    #testset = epairs[0:NTEST] # Try one case
    testset = [testpair]
    templateset = epairs[:i] + epairs[i+1:]
    # itertools will return [(test1, template1), (test1, template2), (test1, template3) ... (testn, templaten)]
    diffs = list(itertools.product(testset, templateset)) 
    transdistance = reg.transdiff(diffs, option='trans', ignoreexception=False, ncore=ncore)
    imagedistance = reg.transdiff(diffs, option='image', ignoreexception=False, ncore=ncore)

    session[i]['testset'] = testset
    session[i]['templateset'] = templateset
    session[i]['transdistance'] = transdistance
    session[i]['imagedistance'] = imagedistance

    with open(join(dbpath, 'expttemplate.pkl'), 'wb') as outfile:
        pickle.dump(session, outfile)


## Prediction
with open('expttemplate.pkl', 'rb') as infile:
    session = pickle.load(infile)

testset = session['testset']
templateset = session['templateset']
transdistance = session['transdistance']
imagedistance = session['imagedistance']

for i, p in enumerate(testset):
    followup = c.find_followups([p], interval=interval)
    followid = followup[0].fixedimage.getimgid()
    #targettemplates = [t for t in dtemplateset if t[0][0] == p]
    #tset = [t[0] for t in targettemplates]
    targettemplateset = templateset
    w = imagedistance[i*len(templateset):(i+1)*len(templateset)]
    imgpreditctionerr = reg.predict(p, templateset, w, real_followupid=followid, ncore=ncore, K=K, outprefix='img')
    w = transdistance[i*len(templateset):(i+1)*len(templateset)]
    transpredictionerr = reg.predict(p, templateset, w, real_followupid=followid, ncore=ncore, K=K, outprefix='trans')
    print 'prediction err is', predictionerr

## Evaluation

