from utils import * 
from os.path import *


# Traverse the dbpath for the files with provided suffix
dbpath = abspath(join('tests', 'testdata', '4092cMCI-GRAPPA2'))
limg = traverse_for_file(dbpath, 'nii')
print('limg:')
print(limg)
assert len(limg) == 5, 'Wrong number of nii files found : %d\n' % len(limg)

mrlist = get_adni_mrlist(dbpath)
assert len(mrlist) == 4
for mr in mrlist:
	filepath = mr.getfilepath()
	print(filepath)
	assert filepath is not None

	# test view slice
	viewslice(filepath)
