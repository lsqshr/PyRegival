from os.path import *
from utils import *
from preprocessing import *
import csv, time
from mrtemplate import *
from mrpredict import *
import time

dbpath = join('tests', "testdata", "5People")
builder = MrTemplateBuilder(dbpath)
ladnimr = get_adni_mrlist(dbpath)

if not os.path.exists(join(dbpath, 'ptemplate.pkl')):
	builder = MrTemplateBuilder(dbpath)
	ptemplate = builder.build(ladnimr, normalise_method='FSL')
else:	
	ptemplate = builder.load_ptemplate(join(dbpath, 'ptemplate.pkl'))

predictor = MrPredictor(dbpath, ptemplate)
targetpair = ('135272', '80317', 12, 12, 24, '311')
predictor.predict(targetpair, real_followup='204131', option='change')