from nipype.interfaces import fsl
from os.path import * 
import os

def betandflirt(lmodel, dbpath):
    bettedpath = join(dbpath, 'betted')

    if not exists(bettedpath):
      os.makedirs(bettedpath)
    
    betnode = fsl.BET()     

    for model in lmodel:
        betnode.inputs.in_file = model.getfilepath()
        betnode.inputs.out_file = join(bettedpath, split(model.getfilepath())[1]+'.betted')
        betnode.run()
