from os.path import join
from utils import *
import csv
from preprocessing import *

dbpath = join('tests', "testdata", "4092cMCI-GRAPPA2")
ladnimr = get_adni_mrlist(dbpath)

betandflirt(ladnimr, dbpath)

bettedfiles = traverse_for_file( join(dbpath, 'betted'), '.nii.gz')
viewslice(join(bettedfiles[0][0], bettedfiles[0][1]))
