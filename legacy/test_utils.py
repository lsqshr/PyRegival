from utils import * 
from os.path import *

''' 
# Test Traverse Files
#################
# Traverse the dbpath for the files with provided suffix
dbpath = abspath(join('tests', 'testdata', '4092cMCI-GRAPPA2'))
limg = traverse_for_file(dbpath, 'nii')
print('limg:')
print(limg)
assert len(limg) == 5, 'Wrong number of nii files found : %d\n' % len(limg)

mrlist = build_adni_mrlist(dbpath)
assert len(mrlist) == 4
for mr in mrlist:
	filepath = mr.getfilepath()
	print(filepath)
	assert filepath is not None

	# test view slice
	viewslice(filepath)
'''

# Test ADNI Collection
##################
# Test Find Transform Pairs
dbpath = join('tests', "testdata", "4092cMCI-GRAPPA2")
ladnimr = build_adni_mrlist(dbpath)
transpairs = AdniMrCollection(ladnimr).find_transform_pairs(interval=[6])
viscodepairs = [ (int(p[1][0].replace('m', '')), int(p[0][0].replace('m', ''))) 
  for p in transpairs]

assert [(0,6), (6,12)] == viscodepairs
print 'Passed: Transform Pairs'
print viscodepairs