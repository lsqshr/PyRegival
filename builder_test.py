from os.path import *
from utils import *
from preprocessing import *
import csv, time
from mrtemplate import *
import time

# Test individual workflows
time1 = time.time()
#dbpath = join('tests', "testdata", "4092cMCI-GRAPPA2")
dbpath = join('tests', "testdata", "5ADNI-Patients")
ladnimr = get_adni_mrlist(dbpath)
time2 = time.time()

# Test Template Builder
builder = MrTemplateBuilder(dbpath)

time3 = time.time()
builder.normalise(ladnimr, normalise_method='FSL') 
time4 = time.time()
builder.normalise(ladnimr, normalise_method='ANTS') 
time5 = time.time()

# Find out the pairs to be transformed [[fixed, moving], ]
transpairs = AdniMrCollection(ladnimr).find_transform_pairs([12,])
builder.transform(ladnimr, transpairs) 
time6 = time.time()

# Test transdiff
diffs = list(itertools.product(transpairs, repeat=2))
diffs = diffs[1:9]
diffs[3] = (diffs[0][0], diffs[0][0])
diffs[5] = (diffs[0][0], diffs[0][0])

# Try to see the order of the output similarity 
# Replace the 3rd and 7th with the same transforms, thus, the similarity should be 1
time7 = time.time()
g = builder.transdiff(diffs)
time8 = time.time()

for node in g.nodes():
    if node.name == 'similarity':
    	s = node.result.outputs.similarity # Tested the order is correct! Congrats
    	print s
    	assert s[2] > 0.98 and s[5] > 0.98

# Test the whole thing all together 
builder = MrTemplateBuilder(dbpath)
time9 = time.time()
ptemplate = builder.build(ladnimr, normalise_method='FSL')
time10 = time.time()

# Make the time log
nsbj = len(ladnimr)
log  = 'Test Time log: %d Subject %d Images\n' % (5, nsbj)
log  += 'Get MR list took\t%f\n' % time2-time1
log  += 'FSL Normalisation took\t%f;\t%f Each\n' % (time2-time1, (time2-time1)/nsbj)
log  += 'FSL Normalisation took\t%f;\t%f Each\n' % (time4-time3, (time4-time3)/nsbj)
log  += 'ANTS Normalisation took\t%f;\t%f Each\n' % (time5-time4, (time5-time4)/nsbj)
log  += 'Inter-subject Transform took\t%f;\t%f Each\n' % (time6-time5, (time6-time5)/(nsbj/2))
log  += 'Transdiff took\t%f;\t%f Each\n' % (time8-time7, (time8-time7)/((nsbj/2)*(nsbj/2)))
log  += 'Whole Template Build Took\t%f\n' % (time10-time9)
print log
with open('test_log.txt', 'w') as f:
	f.write(log)

assert builder.load_ptemplate('ptemplate.pkl') == ptemplate
