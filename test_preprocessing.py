from os.path import *
from utils import *
from preprocessing import *
import csv, time

dbpath = join('tests', "testdata", "4092cMCI-GRAPPA2")
ladnimr = get_adni_mrlist(dbpath)
#normalise(ladnimr, dbpath) 
transform(ladnimr, dbpath, [6,]) 


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