from os.path import *
from utils import *
from preprocessing import *
import csv, time

start = time.time()
dbpath = join('tests', "testdata", "4092cMCI-GRAPPA2")
ladnimr = get_adni_mrlist(dbpath)
preprocess(ladnimr, dbpath) 
