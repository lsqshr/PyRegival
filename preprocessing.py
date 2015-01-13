from nipype.interfaces import fsl
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
from nipype.interfaces.ants.legacy import antsIntroduction
from nipype.interfaces.ants import Registration
from os.path import * 
import os
import multiprocessing
from utils import *

def normalise(lmodel, dbpath, 
                normtemplatepath='/usr/share/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz', 
                normalise_method='ANTS'):
    
    lsbjid = [model.getmetafield('Subject') for model in lmodel]
    limgid = [model.get_imgid() for model in lmodel]
    '''
    inputnode1 = pe.Node(niu.IdentityInterface(fields=['sbjid']), name='input1')
    inputnode1.iterables = ('sbjid', lsbjid)
    '''
    inputnode = pe.Node(niu.IdentityInterface(fields=['imgid']), name='input2')
    inputnode.iterables = ('imgid', limgid)

    # Create a nipype workflow with serial standard_roi => bet => flirt/antNormalization
    datasource = pe.Node(nio.DataGrabber(infields=['imgid'], outfields=['srcimg']), name='niifinder')
    datasource.inputs.base_directory = os.path.abspath(dbpath)
    datasource.inputs.template = '*/*/*/*/ADNI_*_I%s.nii'
    datasource.inputs.sort_filelist = True
    datasource.inputs.template_args['srcimg'] = [['imgid']]

    datasink = pe.Node(nio.DataSink(), name='normsinker')
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
    #wf.connect(inputnode1, 'sbjid', datasource, 'sbjid')
    wf.connect(inputnode, 'imgid', datasource, 'imgid')
    wf.connect(datasource, 'srcimg', stdroi, 'in_file')
    wf.connect(stdroi, 'out_file', stripper, 'in_file')

    if normalise_method == 'FSL':
        normwrap = pe.Node(fsl.FLIRT(bins=640, cost_func='mutualinfo'), name='flirt')
        normwrap.inputs.reference = normtemplatepath
        normwrap.inputs.output_type = "NIFTI_GZ"       #stripper.inputs.padding = True
        infield  = 'in_file'
        outfield = 'out_file'
    elif normalise_method == 'ANTS':
        normwrap = pe.Node(antsIntroduction(), name='ants')
        normwrap.inputs.reference_image = normtemplatepath
        normwrap.inputs.max_iterations = [10, 10, 10]
        normwrap.inputs.transformation_model = 'RA'
        normwrap.inputs.ignore_exception = True
        normwrap.inputs.num_threads = 4
        infield  = 'input_image'
        outfield = 'output_file'

    # The input/output file spec were differnet between FSL and ANTS in nipype
    wf.connect(stripper, 'out_file', normwrap, infield)
    #wf.connect(inputidnode, 'subject_id', datasink, 'container')
    wf.connect(normwrap, outfield, datasink, 'preprocessed')

    # Run workflow with all cpus available
    wf.run(plugin='MultiProc', plugin_args={'n_procs' : multiprocessing.cpu_count()})


def transform(lmodel, dbpath, interval):
    # Find out the pairs to be transformed [[fixed, moving], ]
    transpairs = AdniMrCollection(lmodel).find_transform_pairs(interval)
    
    # Make the workflow
    imgidpairs = [(t[0][1], t[1][1]) for t in transpairs]
    # DEBUG: Only 1 pair
    #imgidpairs = [imgidpairs[0]]

    inputnode = pe.MapNode(interface=niu.IdentityInterface(fields=['fixedimgid', 'movingimgid', 'transid']),
                           name='inputnode',
                           iterfield = ['fixedimgid', 'movingimgid', 'transid'])
    inputnode.inputs.fixedimgid  = [t[0] for t in imgidpairs]
    inputnode.inputs.movingimgid = [t[1] for t in imgidpairs]
    inputnode.inputs.transid = ['%s-%s'% (t[0], t[1]) for t in imgidpairs]

    trans_datasource = pe.MapNode(interface=nio.DataGrabber(
                                         infields=['fixedimgid', 'movingimgid'],
                                         outfields=['fixed_file', 
                                         'moving_file']),
                                  name='trans_datasource', 
                                  iterfield = ['fixedimgid', 'movingimgid'])
    trans_datasource.inputs.base_directory = os.path.abspath(join(dbpath, 'results'))
    #print (trans_datasource.inputs.base_directory)
    trans_datasource.inputs.template = 'preprocessed/_imgid_%s/ants_deformed.nii.gz'
    trans_datasource.inputs.template_args = dict(fixed_file=[['fixedimgid']],
                                                 moving_file=[['movingimgid']])
    trans_datasource.inputs.sort_filelist = True

    transnode = pe.MapNode(interface=SynQuick(),
                           name='transnode',
                           iterfield=['fixed_image', 'moving_image'])
    transnode.inputs.output_prefix = 'out'
    transnode.inputs.dimension = 3 
    #transnode = make_antsRegistrationNode() # in nipype antsRegistration has a non-valid flag: 
                                             # `--collapse-linear-transforms-to-fixed-image-header` 
                                             # which cannot be turned off till 10/1/15

    datasink = pe.MapNode(nio.DataSink(infields=['container',
                                                 'SyNQuick', 
                                                 'SyNQuick.@inverse_warp',
                                                 'SyNQuick.@affine',
                                                 'SyNQuick.@inverse_warped_image',
                                                 'SyNQuick.@warped_image']), 
                          iterfield=['container',
                                     'SyNQuick', 
                                     'SyNQuick.@inverse_warp',
                                     'SyNQuick.@affine',
                                     'SyNQuick.@inverse_warped_image',
                                     'SyNQuick.@warped_image'], name='transsinker')
    datasink.inputs.base_directory = os.path.abspath(join(dbpath, 'results', 'transformed'))
    datasink.inputs.substitutions = [('_transnode', 'transid')]
    datasink.inputs.parameterization = True

    # build the workflow
    wf = pe.Workflow(name="transform")
    wf.connect(inputnode, 'fixedimgid', trans_datasource, 'fixedimgid')
    wf.connect(inputnode, 'movingimgid', trans_datasource, 'movingimgid')
    wf.connect(trans_datasource, 'fixed_file', transnode, 'fixed_image')
    wf.connect(trans_datasource, 'moving_file', transnode, 'moving_image')
    wf.connect(inputnode, 'transid', datasink, 'container')
    wf.connect(transnode, 'warp_field', datasink, 'SyNQuick')
    wf.connect(transnode, 'inverse_warp_field', datasink, 'SyNQuick.@inverse_warp')
    wf.connect(transnode, 'affine_transformation', datasink, 'SyNQuick.@affine')
    wf.connect(transnode, 'inverse_warped', datasink, 'SyNQuick.@inverse_warped_image')
    wf.connect(transnode, 'warped_image', datasink, 'SyNQuick.@warped_image')

    # Run workflow with all cpus available
    wf.run(plugin='MultiProc', plugin_args={'n_procs' : multiprocessing.cpu_count()})


# Make fillin the default values
# Currently unused because of the invalid flag passed by nipype
def make_antsRegistrationNode():
    reg = Registration()
    reg.inputs.output_transform_prefix = "output_"
    #reg.inputs.initial_moving_transform = 'trans.mat'
    #reg.inputs.invert_initial_moving_transform = False 
    reg.inputs.transforms = ['Affine', 'SyN']
    reg.inputs.transform_parameters = [(2.0,), (0.25, 3.0, 0.0)]
    reg.inputs.number_of_iterations = [[1500, 200], [100, 50, 30]]
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = False
    reg.inputs.metric = ['Mattes']*2
    reg.inputs.metric_weight = [1]*2 # Default (value ignored currently by ANTs)
    reg.inputs.radius_or_number_of_bins = [32]*2
    reg.inputs.sampling_strategy = ['Random', None]
    reg.inputs.sampling_percentage = [0.05, None]
    reg.inputs.convergence_threshold = [1.e-8, 1.e-9]
    reg.inputs.convergence_window_size = [20]*2
    reg.inputs.smoothing_sigmas = [[1,0], [2,1,0]]
    reg.inputs.sigma_units = ['vox'] * 2
    reg.inputs.shrink_factors = [[2,1], [3,2,1]]
    reg.inputs.use_estimate_learning_rate_once = [True, True]
    reg.inputs.use_histogram_matching = [True, True] # This is the default
    return pe.Node(reg,'transnode')
