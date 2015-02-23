from os.path import *
import os
from models import *
import csv, time
from regival import *
import time
import itertools
import numpy as np
import pickle
 
NTEST = 1
interval = [12]
dbpath = '/media/siqi/SiqiLarge/ADNI-For-Pred'
ncore = 4
DECAY = 0.9
WEIGHTING = 'ALL'
CROSSW = 0.7
LONGW = 0.3
K = 5
 
c = AdniMrCollection(dbpath=dbpath, regendb=False)
#c.filtermodels(interval=[12])
#c.randomselect(60, interval)
 
## Template Construction
reg = MrRegival(collection=c, dbpath=dbpath)
epairs = reg.getcollection().filter_elligible_pairs(interval=interval)
# Without cleaning transformed subjects would have more than one transid
#reg.normalise(ignoreexception=True, ncore=ncore)
#reg.transform(ignoreexception=True, ncore=ncore)
#reg.cross_jacdet(ncore=ncore, ignoreexception=False)
TRIALSTART = 0
TRIALEND = 10
'''
 
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
    if session[i] == 0:
        session[i] = {}
    #testset = epairs[0:NTEST] # Try one case
    testset = [testpair]
    templateset = epairs[:i] + epairs[i+1:]
    # Note: itertools will return [(test1, template1), (test1, template2), (test1, template3) ... (testn, templaten)]
    diffs = list(itertools.product(testset, templateset)) 
    #transdistance = reg.transdiff(diffs, option='trans', ignoreexception=False, ncore=ncore)
    imagedistance = reg.transdiff(diffs, option='image', ignoreexception=False, ncore=ncore)
    #longjdistance = reg.transdiff(diffs, option='longitudinal_jacobian', ignoreexception=False, ncore=ncore)
    #crossjdistance = reg.transdiff(diffs, option='crosssectional_jacobian', ignoreexception=False, ncore=ncore)
 
    session[i]['testset'] = testset
    session[i]['templateset'] = templateset
    #session[i]['transdistance'] = transdistance
    session[i]['imagedistance'] = imagedistance
    #session[i]['longjdistance'] = longjdistance
    #session[i]['crossjdistance'] = crossjdistance
 
    with open(join(dbpath, 'expttemplate.pkl'), 'wb') as outfile:
        pickle.dump(session, outfile)
'''

## Prediction
with open(join(dbpath, 'expttemplate.pkl'), 'rb') as infile:
    session = pickle.load(infile)
 
for i, testpair in enumerate(epairs):
    #if i < TRIALSTART:
    #    continue
    if i < TRIALSTART:
        continue
    if i > TRIALEND:
        break;
    
    testset = session[i]['testset']
    templateset = session[i]['templateset']
    #transdistance = session[i]['transdistance']
    imagedistance = session[i]['imagedistance']
    longjddistance = session[i]['longjdistance']
    crossjddistance = session[i]['crossjdistance']
 
    for j, p in enumerate(testset):
        longw = longjddistance[j*len(templateset):(j+1)*len(templateset)]
        crossw = crossjddistance[j*len(templateset):(j+1)*len(templateset)]
        imgw = imagedistance[j*len(templateset):(j+1)*len(templateset)]
        mergew = list(CROSSW * np.array(longw) + LONGW * np.array(crossw))

        if WEIGHTING in ['CROSS', 'ALL']:
            reg.predict(p, templateset, crossw, decayratio=DECAY, ncore=ncore, K=K, outprefix='crosssectional_jacobian')
        if WEIGHTING in ['LONG', 'ALL']:
            reg.predict(p, templateset, longw, decayratio=DECAY, ncore=ncore, K=K, outprefix='longitudinal_jacobian')
        if WEIGHTING in ['MERGE', 'ALL']:
            mergepredictionerr = reg.predict(p, templateset, mergew, decayratio=DECAY, ncore=ncore, K=K, outprefix='merge')
        if WEIGHTING in ['IMAGE', 'ALL']:
            mergepredictionerr = reg.predict(p, templateset, imgw, decayratio=DECAY, ncore=ncore, K=K, outprefix='image')

'''
## Evaluation
testset = session[0].testset + session[0].templateset
limgerr = []
ltranserr = []
lmergeerr = []

for i, p in enumerate(testset):
    [origerr, regerr, rawerr] = reg.evaluate(p, ncore=ncore, outprefix='crosssectional_jacobian')
    print 'CROSS Err:', oerr, 'Registered Err:', rerr, 'Raw Err:', rawerr
    limgerr.append([origerr, regerr, rawerr])
 
    [origerr, regerr, rawerr] = reg.evaluate(p, ncore=ncore, outprefix='longitudinal_jacobian')
    print 'LONG Err:', oerr, 'Registered Err:', rerr, 'Raw Err:', rawerr
    ltranserr.append([origerr, regerr, rawerr])
 
    [origerr, regerr, rawerr] = reg.evaluate(p, ncore=ncore, outprefix='merge')
    print 'Merge Original Err:', oerr, 'Registered Err:', rerr, 'Raw Err:', rawerr
    lmergeerr.append([origerr, regerr, rawerr])
 
err = {}
err['limgerr'] = limgerr
err['ltranserr'] = ltranserr
#err['lmergeerr'] = lmergeerr
 
with open('predicterr.pkl', 'wb') as f:
    pickle.dump(err, f)
'''
