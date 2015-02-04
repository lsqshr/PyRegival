from os.path import *
import os
from models import *
import csv, time
from regival import *
import time

dbpath = '/media/siqi/SiqiLarge/ADNI-For-Pred'
ncore = 4

c = AdniMrCollection(dbpath=dbpath, regendb=False)
c.randomselect(60, [12])

reg = MrRegival(collection=c, dbpath=dbpath)
reg.normalise(ignoreexception=True, ncore=ncore)
reg.transform(ignoreexception=True, ncore=ncore)
reg.transdiff(ignoreexception=True, ncore=2)
