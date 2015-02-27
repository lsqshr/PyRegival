from os.path import *
import os
import sys
from models import *
import csv, time
from regival import *
import time
import itertools
import numpy as np
import pickle
 
NTEST = 1
interval = [12]
dbpath = '/home/siqi/ADNI-For-Pred'
ncore = 60 
DECAY = 0.9
WEIGHTING = 'IMGSORT'
CROSSW = 0.3
LONGW = 0.7
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
if len(sys.argv) == 1:
    TRIALSTART = 0
    TRIALEND = 61
else:
    TRIALSTART = int(sys.argv[1])
    TRIALEND = int(sys.argv[2])


'''
session = [0] * len(epairs)
# Create Leave-one-out Sets
for i, testpair in enumerate(epairs):
    session[i] = {}
    testset = [testpair]
    templateset = epairs[:i] + epairs[i+1:]
    session[i]['testset'] = testset
    session[i]['templateset'] = templateset
with open(join(dbpath, 'expttemplate.pkl'), 'wb') as outfile:
    pickle.dump(session, outfile)

'''
with open(join(dbpath, 'expttemplate.pkl'), 'rb') as infile:
    session = pickle.load(infile)

'''
# Collect Leave one out diff
for i, testpair in enumerate(session):
    if i < TRIALSTART:
        continue
    if i > TRIALEND:
        break;

    # session will be save after every leave-one-out trial
    tempsession = session[i]
    testset = tempsession['testset']
    templateset = tempsession['templateset']
    # Note: itertools will return [(test1, template1), (test1, template2), (test1, template3) ... (testn, templaten)]
    diffs = list(itertools.product(testset, templateset)) 
    #transdistance = reg.transdiff(diffs, option='trans', ignoreexception=False, ncore=ncore)
    imagedistance = reg.transdiff(diffs, option='image', ignoreexception=False, ncore=ncore)
    longjdistance = reg.transdiff(diffs, option='longitudinal_jacobian', ignoreexception=False, ncore=ncore)
    crossjdistance = reg.transdiff(diffs, option='crosssectional_jacobian', ignoreexception=False, ncore=ncore)
 
    tempsession['imagedistance'] = imagedistance
    tempsession['longjdistance'] = longjdistance
    tempsession['crossjdistance'] = crossjdistance
 
    with open(join(dbpath, 'expttemplate%d.pkl'%i), 'wb') as outfile:
        pickle.dump(tempsession, outfile)

'''
'''
# Collect all sessions
for i in range(len(session)):
    print 'collecting results'
    if i < TRIALSTART:
        continue
    if i > TRIALEND:
        break;
    with open(join(dbpath, 'expttemplate%d.pkl'%i), 'rb') as infile:
        tempsession = pickle.load(infile)
    for key in tempsession:
        session[i][key] = tempsession[key]
with open(join(dbpath, 'expttemplate.pkl'), 'wb') as outfile:
    pickle.dump(session, outfile)
'''

'''
## Prediction
 
for i, testpair in enumerate(session):
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
	mergeimagew = list(0.4 * np.array(mergew) + 0.6 * np.array(imgw))

        if WEIGHTING in ['CROSS', 'ALL']:
            reg.predict(p, templateset, crossw, decayratio=DECAY, ncore=ncore, K=K, outprefix='crosssectional_jacobian')
        if WEIGHTING in ['LONG', 'ALL']:
            reg.predict(p, templateset, longw, decayratio=DECAY, ncore=ncore, K=K, outprefix='longitudinal_jacobian')
        if WEIGHTING in ['MERGE', 'ALL']:
            mergepredictionerr = reg.predict(p, templateset, mergew, decayratio=DECAY, ncore=ncore, K=K, outprefix='merge')
        if WEIGHTING in ['IMAGE', 'ALL']:
            mergepredictionerr = reg.predict(p, templateset, imgw, decayratio=DECAY, ncore=ncore, K=K, outprefix='image')
        if WEIGHTING in ['IMGSORT', 'ALL']:
            mergepredictionerr = reg.predict(p, templateset, mergew, decayratio=DECAY, ncore=ncore, K=K, outprefix='imgsort', sortweights=imgw)

'''
## Evaluation
ldir = os.listdir(join(dbpath, 'results', 'imagepredicted'))

testset = session[0]['testset'] + session[0]['templateset']
crosserr = []
longerr = []
mergeerr = []
imageerr = []
imgsorterr = []

for i, p in enumerate(testset):
    if p.movingimage.getimgid() + '-' + p.fixedimage.getimgid() not in ldir:
        continue

    [origerr, regerr, rawerr] = reg.evaluate(p, ncore=ncore, outprefix='crosssectional_jacobian')
    crosserr.append([origerr, regerr, rawerr])
 
    [origerr, regerr, rawerr] = reg.evaluate(p, ncore=ncore, outprefix='longitudinal_jacobian')
    longerr.append([origerr, regerr, rawerr])
 
    [origerr, regerr, rawerr] = reg.evaluate(p, ncore=ncore, outprefix='merge')
    mergeerr.append([origerr, regerr, rawerr])

    [origerr, regerr, rawerr] = reg.evaluate(p, ncore=ncore, outprefix='image')
    imageerr.append([origerr, regerr, rawerr])

    [origerr, regerr, rawerr] = reg.evaluate(p, ncore=ncore, outprefix='imgsort')
    imgsorterr.append([origerr, regerr, rawerr])

 
err = {}
err['crosserr'] = crosserr
err['longerr'] = longerr
err['mergeerr'] = mergeerr
err['imageerr'] = imageerr
err['imgsorterr'] = imgsorterr

with open('predicterr.pkl', 'wb') as f:
    pickle.dump(err, f)

## Plot Err
figsize = (16,8)
dpi = 200

from pylab import *
import matplotlib.pyplot as plt
import matplotlib

with open('predicterr.pkl', 'rb') as f:
    err = pickle.load(f)

# Plot three lines of merge
matplotlib.rcParams.update({'font.size': 15})
mergeerr = np.array(err['mergeerr'])
sortidx = np.argsort(mergeerr[:, 2])
sortedmergeerr = mergeerr[sortidx, :]

fig1, ax1 = subplots(figsize=figsize, dpi=dpi)
ax1.plot(np.squeeze(sortedmergeerr[:,0]), label='P-B', ls=':')
ax1.plot(np.squeeze(sortedmergeerr[:,1]), label='P-rA', ls='-')
ax1.plot(np.squeeze(sortedmergeerr[:,2]), color='black', label='REAL', marker='+', ls='-')
ax1.legend(loc='lower right')
ax1.set_yscale('log')
#ax1.set_title('3 Types of Errors')
fig1.savefig('3err-merge.eps')


# Plot three lines of cross 
mergeerr = np.array(err['crosserr'])
sortidx = np.argsort(mergeerr[:, 2])
sortedmergeerr = mergeerr[sortidx, :]

fig1, ax1 = subplots()
ax1.plot(np.squeeze(sortedmergeerr[:,0]), label='P-B', ls=':')
ax1.plot(np.squeeze(sortedmergeerr[:,1]), label='P-rA', ls='-')
ax1.plot(np.squeeze(sortedmergeerr[:,2]), color='black', label='REAL', marker='+', ls='-')
ax1.legend(loc='lower right')
ax1.set_yscale('log')
ax1.set_title('3 Types of Errors of Cross-sectional Weights')
#fig1.show()
fig1.savefig('3errors-cross.eps')

# Plot three lines of long
mergeerr = np.array(err['longerr'])
sortidx = np.argsort(mergeerr[:, 2])
sortedmergeerr = mergeerr[sortidx, :]

fig1, ax1 = subplots()
ax1.plot(np.squeeze(sortedmergeerr[:,0]), label='P-B', ls=':')
ax1.plot(np.squeeze(sortedmergeerr[:,1]), label='P-rA', ls='-')
ax1.plot(np.squeeze(sortedmergeerr[:,2]), color='black', label='REAL', marker='+', ls='-')
ax1.legend(loc='lower right')
ax1.set_yscale('log')
ax1.set_title('3 Types of Errors of Longitudinal Weights')
#fig1.show()
fig1.savefig('3errors-long.eps')

# Plot oerr over four methods

crosserr = np.array(err['crosserr'])[:,0]
longerr = np.array(err['longerr'])[:,0]
mergeerr = np.array(err['mergeerr'])[:,0]
imageerr = np.array(err['imageerr'])[:,0]
imgsorterr = np.array(err['imgsorterr'])[:,0]
rawerr = np.array(err['imageerr'])[:, 2]

sortidx = np.argsort(rawerr)
sortedcrosserr = crosserr[sortidx]
sortedlongerr = longerr[sortidx]
sortedmergeerr = mergeerr[sortidx]
sortedimageerr = imageerr[sortidx]
sortedimgsorterr = imgsorterr[sortidx]
sortedrawerr = rawerr[sortidx]

fig2, ax2 = subplots(figsize=figsize, dpi=dpi)
ax2.plot(np.squeeze(sortedcrosserr), label='cross', ls=':')
ax2.plot(np.squeeze(sortedlongerr), label='long', ls='-.')
ax2.plot(np.squeeze(sortedmergeerr), label='merge', ls='-', marker='o')
ax2.plot(np.squeeze(sortedimageerr), label='intensity', ls=':', color='purple', marker='x')
ax2.plot(np.squeeze(sortedimgsorterr), label='imgsort', ls=':', color='black', marker='^')
#ax2.plot(np.squeeze(sortedmergeimageerr), label='mergeimage+intensity', ls='-', color='purple', marker='o')
#ax2.plot(np.squeeze(sortedimagenocuterr), label='intensity-nocutoff', ls='-', color='purple', marker='o')
#ax2.plot(np.squeeze(sortedimageerr), color='black', label='intensity', ls=':')
ax2.plot(np.squeeze(sortedrawerr), color='black', label='REAL', marker='+', ls='-')
ax2.set_yscale('log')
ax2.legend(loc='lower right')
#ax2.set_title('P-B three methods')
#fig2.show()
fig2.savefig('p-b.eps')

## Plot rerr over four methods
crosserr = np.array(err['crosserr'])[:,1]
longerr = np.array(err['longerr'])[:,1]
mergeerr = np.array(err['mergeerr'])[:,1]
imageerr = np.array(err['imageerr'])[:,1]
imgsorterr = np.array(err['imgsorterr'])[:,1]
rawerr = np.array(err['imageerr'])[:, 2]
sortidx = np.argsort(rawerr)
sortedcrosserr = crosserr[sortidx]
sortedlongerr = longerr[sortidx]
sortedmergeerr = mergeerr[sortidx]
sortedimageerr = imageerr[sortidx]
sortedimgsorterr = imgsorterr[sortidx]
sortedrawerr = rawerr[sortidx]

fig3, ax3 = subplots(figsize=figsize, dpi=dpi)
ax3.plot(np.squeeze(sortedcrosserr), label='cross', ls=':')
ax3.plot(np.squeeze(sortedlongerr), label='long', ls='-.')
ax3.plot(np.squeeze(sortedmergeerr), label='merge', ls='-', marker='o')
ax3.plot(np.squeeze(sortedimageerr), label='intensity', ls=':', color='purple', marker='x')
ax3.plot(np.squeeze(sortedimageerr), label='imgsort', ls=':', color='black', marker='^')
ax3.plot(np.squeeze(sortedrawerr), color='black', label='REAL', marker='+', ls='-')
ax3.legend(loc='upper left')
#ax3.set_title('P-rA three methods')
ax3.set_yscale('log')
#fig3.show()
fig3.savefig('p-ra.eps')

