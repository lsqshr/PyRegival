from os.path import *
from utils import *
from preprocessing import *
import csv, time
from regival import *


#dbpath = join('tests', "testdata", "4092cMCI-GRAPPA2")
dbpath = join('tests', "testdata", "5ADNI-Patients")
ladnimr = get_adni_mrlist(dbpath)

normalise(ladnimr, dbpath, normalise_method='FSL') 
# Find out the pairs to be transformed [[fixed, moving], ]
transpairs = AdniMrCollection(ladnimr).find_transform_pairs([12,])
transform(ladnimr, dbpath, transpairs) 

'''
movingimg = '/home/siqi/workspace/ContentBasedRetrieval/PyRegival/tests/testdata/4092cMCI-GRAPPA2/116_S_4092/MPRAGE_GRAPPA2/2011-06-24_09_00_15.0/S112543/ADNI_116_S_4092_MR_MPRAGE_GRAPPA2_br_raw_20110624151135946_18_S112543_I241691.nii'
fixedimg = '/home/siqi/workspace/ContentBasedRetrieval/PyRegival/tests/testdata/4092cMCI-GRAPPA2/116_S_4092/MPRAGE_GRAPPA2/2012-01-13_12_01_11.0/S137148/ADNI_116_S_4092_MR_MPRAGE_GRAPPA2_br_raw_20120118092234173_73_S137148_I278831.nii'

trans = pe.Node(interface=SynQuick(), name='transform')
trans.inputs.fixed_image = [fixedimg]
trans.inputs.moving_image= [movingimg]
trans.inputs.dimension = 3
trans.inputs.output_prefix = '/home/siqi/Desktop/out'
trans.run()
'''

# Test transdiff
diffs = list(itertools.product(transpairs, repeat=2))
diffs = diffs[1:9]
diffs[3] = (diffs[0][0], diffs[0][0])
diffs[5] = (diffs[0][0], diffs[0][0])

# Try to see the order of the output similarity 
# Replace the 3rd and 7th with the same transforms, thus, the similarity should be 1
g = transdiff(dbpath, diffs)

for node in g.nodes():
    if node.name == 'similarity':
    	s = node.result.outputs.similarity # Tested the order is correct! Congrats
    	print s
    	assert s[2] > 0.98 and s[5] > 0.98

