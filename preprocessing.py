from nipype.interfaces import fsl
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
from nipype.interfaces.ants.legacy import antsIntroduction
from os.path import * 
import os
import multiprocessing
from utils import StdRoi

def preprocess(lmodel, dbpath, 
               normtemplatepath='/usr/share/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz', 
               normalise_method='ANTS'):
    '''
    bettedpath = join(dbpath, 'betted')

    if not exists(bettedpath):
      os.makedirs(bettedpath)
    '''

    # Create a nipype workflow with serial standard_roi => bet => flirt/antNormalization
    inputnode  = pe.Node(interface=niu.IdentityInterface(fields=['in_file']), name = 'inputspec')
    inputidnode  = pe.Node(interface=niu.IdentityInterface(fields=['subject_id']), name = 'inputidspec')
    outputnode = pe.Node(interface=niu.IdentityInterface(fields=['out_file']), name = 'outputspec')

    inputnode.iterables = ('in_file', [model.getfilepath() for model in lmodel])
    #inputidnode.iterables = ('subject_id', [model.get_imgid() for model in lmodel])

    ''' 
    # Replaced outputnode to DataSink in nipype: we did not use the DataGrabber because ADNI
    #                                           filename template is way to compilcated

    outputnode.iterables = ('out_file', \
                            [join(dbpath, 'flirted', split(model.getfilepath())[-1])\
                             for model in lmodel])
    '''

    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = os.path.abspath(join(dbpath, 'results'))

    # Start to make build the workflow
    wf = pe.Workflow(name="preprocess")
 
    """
    Estimate the tissue classes from the anatomical image. But use spm's segment
    as FSL appears to be breaking.
    """
    stdroi = pe.Node(StdRoi(), name='standard_space_roi')
    stdroi.inputs.betpremask = True
    stripper = pe.Node(fsl.BET(), name='stripper')
    stripper.inputs.frac = 0.2

    #stripper.inputs.robust = True
    #stripper.inputs.reduce_bias = True
    wf.connect(inputnode, 'in_file', stdroi, 'in_file')
    wf.connect(stdroi, 'out_file', stripper, 'in_file')

    if normalise_method == 'FSL':
        normwrap = pe.Node(fsl.FLIRT(bins=640, cost_func='mutualinfo'), name='flirt')
        normwrap.inputs.reference = normtemplatepath
        normwrap.inputs.output_type = "NIFTI_GZ"       #stripper.inputs.padding = True
        infield = 'in_file'
        outfield = 'out_file'
    elif normalise_method == 'ANTS':
        normwrap = pe.Node(antsIntroduction(), name='ants')
        normwrap.inputs.reference_image = normtemplatepath
        normwrap.inputs.max_iterations = [10, 10, 10]
        normwrap.inputs.transformation_model = 'RA'
        normwrap.inputs.ignore_exception = True
        normwrap.inputs.num_threads = 4 
        infield = 'input_image'
        outfield = 'output_file'

    # The input/output file spec were differnet between FSL and ANTS in nipype
    wf.connect(stripper, 'out_file', normwrap, infield)
    #wf.connect(inputidnode, 'subject_id', datasink, 'container')
    wf.connect(normwrap, outfield, datasink, 'preprocessed')

    # Run workflow with all cpus available
    wf.run(plugin='MultiProc', plugin_args={'n_procs' : multiprocessing.cpu_count()})
