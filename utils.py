import os
import subprocess
import re
from os.path import * 
import csv
from models import adnimrimg
from nipype.interfaces.fsl.preprocess import FSLCommandInputSpec, FSLCommand
from nipype.interfaces.ants.registration import ANTSCommand, ANTSCommandInputSpec, RegistrationOutputSpec
from nipype.interfaces.base import TraitedSpec, File, traits, CommandLine, InputMultiPath
from nipype.interfaces.traits_extension import isdefined

# Traverse the dbpath for the files with provided suffix
def traverse_for_file(path, suffix):
    limg = [];
    for root, dirs, files in os.walk(path):
      for file in files:
        if file.endswith(suffix):
            limg.append((root, file))
    return limg


# Find Image ID from ADNI nii filename
def find_adni_imgid(fname):
    lmatch = re.findall('_I\d+', fname)
    assert len(lmatch) <= 1 , 'More than one matches were found: '
    cleanmatch = [] 

    for m in lmatch: 
      cleanmatch.append(m[2:]) # Remove the '_' before image id 

    return cleanmatch[0] if len(cleanmatch) == 1 else None


def get_adni_mrlist(dbpath):
    # Find all images stated in the dbgen.csv in the db path
    # Construct adnimrimg list
    limg = traverse_for_file(dbpath, '.nii')
    limgid = [find_adni_imgid(img[1]) for img in limg ]
    ladnimr = []

    # read in the dbgen.db
    with open(join(dbpath, 'dbgen.csv')) as f:
      r = csv.reader(f, delimiter=',')
      header = next(r)
      #imgididx = header.index('Image.Data.ID')

      for row in r:
        meta = {}
        for i, col in enumerate(row):
            meta[header[i]] = col

        imgid = meta['Image.Data.ID']
        print('=====limgid==============')
        print(limgid) 
        print('===================')

        imgf = limg[ limgid.index(imgid) ]
        filepath = join(imgf[0], imgf[1])
        ladnimr.append(adnimrimg(meta, filepath))

    return ladnimr

def viewslice(file):
    path, fname = split(file)
    subprocess.Popen(["tkmedit",'-f', file])

# A collection tool class to perform some grouping and searching of the adnimriimg models
class AdniMrCollection(object):
    def __init__(self, lmodel):
        self.lmodel = lmodel 

    # Return Image.Data.ID pairs [[fixed_img_id1, moving_img_id1], [fixed_img_id2, moving_img_id2],...]
    def find_transform_pairs(self, interval=[6,12]):
        sbjdict = self.group_sbj()
        transpairs = []

        # Find the intervals with the specific lengths
        for key, sbj_imglist in sbjdict.iteritems():
            #sbj_imglist = sbjdict[key]
            sbj_imglist.sort(key=lambda x: x[0])

            print [ sbj[0] for sbj in sbj_imglist ] # Check the sorting

            for i in xrange(len(sbj_imglist)):
                for j in xrange(i+1,len(sbj_imglist)):
                    viscode1 = int(sbj_imglist[i][0].replace('m', ''))
                    viscode2 = int(sbj_imglist[j][0].replace('m', ''))
                    if (viscode2 - viscode1) in interval:
                        transpairs.append((sbj_imglist[j], sbj_imglist[i]))

        return transpairs

    # Group models into an RID dictionary <RID, [(VISCODE1, Image.Data.ID1), ...]> 
    def group_sbj(self):
        sbjdict = {}
        for model in self.lmodel:
            rid     = model.getmetafield('RID') 
            viscode = model.getmetafield('VISCODE')
            imgid   = model.getmetafield('Image.Data.ID')
            if rid in sbjdict:
                sbjdict[rid].append((viscode, imgid))
            else:
                sbjdict[rid] = [(viscode, imgid)]

        return sbjdict


class StdRoiInputSpec(FSLCommandInputSpec): 
    in_file  = File(exists=True, 
                    desc = 'path/name of the image to be masked',
                    argstr='%s', position=0, mandatory=True)
    out_file = File(argstr='%s', desc='masked output file',
                    name_source=['in_file'], name_template='%s_roi',
                    position=1, hash_files=False)
    betpremask = traits.Bool(desc='create surface outline image',
                    argstr='-b', position=2)

class StdRoiOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='path/name of the masked image')

#------------------
# Wrap the 'standard_space_roi' of FSL, did not find this interface in the current nipype
class StdRoi(FSLCommand):
    _cmd = 'standard_space_roi'
    input_spec = StdRoiInputSpec
    output_spec = StdRoiOutputSpec

class SynQuickInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='-d %d', usedefault=False,
                            position=1, desc='image dimension (2 or 3)')
    fixed_image = InputMultiPath(File(exists=True), mandatory=True, argstr='-f %s',
                                    desc=('image to apply transformation to (generally a coregistered '
                                            'functional)'))
    moving_image = InputMultiPath(File(exists=True), argstr='-m %s',
                                    mandatory=True,
                                    desc=('image to apply transformation to (generally a coregistered '
                                            'functional)'))
    output_prefix = traits.Str('out', usedefault=True,
                                            argstr='-o %s',
                                            mandatory=True, desc='')

class SynQuickOutputSpec(ANTSCommandInputSpec):
    warp_field = File(exists=True, desc='warped displacement fields')
    inverse_warp_field = File(exists=True, desc='inversed warp displacement fields')
    affine_transformation = File(exists=True, desc='affine pre-registration matrix')
    inverse_warped = File(exists=True, desc='resampled inverse image')
    warped_image = File(exists=True, desc='resampled forward image')

class SynQuick(ANTSCommand):
    _cmd = 'antsRegistrationSyNQuick.sh'
    input_spec = SynQuickInputSpec 
    output_spec = SynQuickOutputSpec

    def _list_outputs(self): # Adapted from ants.registration.Registration
        outputs = self._outputs().get()

        outputs['warp_field'] = os.path.join(os.getcwd(),
                                 self.inputs.output_prefix +
                                 '1Warp.nii.gz')
        outputs['inverse_warp_field'] = os.path.join(os.getcwd(),
                                 self.inputs.output_prefix +
                                     '1InverseWarp.nii.gz')

        outputs['affine_transformation'] = os.path.join(os.getcwd(),
                                                        self.inputs.output_prefix +
                                                        '0GenericAffine.mat')
        outputs['inverse_warped'] = os.path.join(os.getcwd(),
                                             self.inputs.output_prefix +
                                             'InverseWarped.nii.gz')
        outputs['warped_image'] = os.path.join(os.getcwd(),
                                              self.inputs.output_prefix +
                                              'Warped.nii.gz')
        return outputs