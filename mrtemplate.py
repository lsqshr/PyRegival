from nipype.interfaces import fsl
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
from nipype.interfaces.ants.legacy import antsIntroduction
from nipype.interfaces.ants import Registration
from nipype.interfaces.ants import ApplyTransforms
from nipype.algorithms.metrics import Similarity
from os.path import * 
import os
import multiprocessing
from utils import *
from additional_nipype_interfaces import *
import itertools
import pickle

class MrTemplateBuilder (object): 

    def __init__(self, dbpath):
        self.dbpath = dbpath

    def build(self, lmodel, normtemplatepath='/usr/share/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz', 
                       normalise_method='ANTS',
                       intervals=[12]):
        '''
        lmodel: a list of adnimrimg objects
        normtemplatepath: the template used by the normalisation

        '''
        self.normalise(lmodel, normtemplatepath, normalise_method)
        transpairs = AdniMrCollection(lmodel).find_transform_pairs(intervals)
        self.transform(lmodel, transpairs)
        diffs = list(itertools.product(transpairs, repeat=2))
        # Make diffs to a {fid1,mid1,fid2,mid2: (interval1, interval2)} dict
        # Structure of diff itself: ((fid1,mid1, interval1), (fid2,mid2, interval2))
        diffintervals = {}
        for d in diffs:
            diffintervals[d[0][0].get_imgid(), d[0][1].get_imgid(),
                               d[1][0].get_imgid(), d[1][1].get_imgid()] = (d[0][2], d[1][2])

        g = self.transdiff(diffs) # transdiff does not consider the different intervals

        for node in g.nodes():
            if node.name == 'similarity':
                similarity = node.result.outputs.similarity # The order is the same witht self.diffs

        pred_template = {}
        pred_template['diff_intervals'] = diffintervals
        pred_template['transpairs'] = transpairs
        pred_template['corr'] = dict(zip(diffs, similarity)) 
        pred_template['lmodel'] = lmodel 

        with open('ptemplate.pkl', 'wb') as outfile:
            pickle.dump(pred_template, outfile)

        return pred_template


    def load_ptemplate(self, filepath):
        with open(filepath, 'rb') as infile:
            ptemplate = pickle.load(infile)
        return ptemplate


    def normalise(self, lmodel, 
                normtemplatepath='/usr/share/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz', 
                normalise_method='ANTS'):
        ''' 

        '''
        
        lsbjid = [model.getmetafield('Subject') for model in lmodel]
        limgid = [model.get_imgid() for model in lmodel]
        inputnode = pe.Node(niu.IdentityInterface(fields=['imgid']), name='input2')
        inputnode.iterables = ('imgid', limgid)

        # Create a nipype workflow with serial standard_roi => bet => flirt/antNormalization
        datasource = pe.Node(nio.DataGrabber(infields=['imgid'], outfields=['srcimg']), name='niifinder')
        datasource.inputs.base_directory = os.path.abspath(self.dbpath)
        datasource.inputs.template = '*/*/*/*/ADNI_*_I%s.nii'
        datasource.inputs.sort_filelist = True
        datasource.inputs.template_args['srcimg'] = [['imgid']]

        datasink = pe.Node(nio.DataSink(), name='normsinker')
        datasink.inputs.base_directory = os.path.abspath(join(self.dbpath, 'results'))

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
            normwarp = pe.Node(fsl.FLIRT(bins=640, cost_func='mutualinfo'), name='flirt')
            normwarp.inputs.reference = normtemplatepath
            normwarp.inputs.output_type = "NIFTI_GZ"       #stripper.inputs.padding = True
            normwarp.inputs.out_file = 'norm_deformed.nii.gz'
            infield  = 'in_file'
            outfield = 'out_file'
        elif normalise_method == 'ANTS':
            normwarp = pe.Node(antsIntroduction(), name='ants')
            normwarp.inputs.reference_image = normtemplatepath
            normwarp.inputs.max_iterations = [30,90,20]
            normwarp.inputs.transformation_model = 'RA'
            normwarp.inputs.out_prefix = 'norm_'
            infield  = 'input_image'
            outfield = 'output_file'

        # The input/output file spec were differnet between FSL and ANTS in nipype
        wf.connect(stripper, 'out_file', normwarp, infield)
        #wf.connect(inputidnode, 'subject_id', datasink, 'container')
        wf.connect(normwarp, outfield, datasink, 'preprocessed')

        # Run workflow with all cpus available
        wf.run(plugin='MultiProc', plugin_args={'n_procs' : multiprocessing.cpu_count()})


    def transform(self, lmodel, transpairs):
            
        # Make the workflow
        inputnode = pe.MapNode(interface=niu.IdentityInterface(fields=['fixedimgid', 'movingimgid', 'transid']),
                               name='inputnode',
                               iterfield = ['fixedimgid', 'movingimgid', 'transid'])
        inputnode.inputs.fixedimgid  = [t[0].get_imgid() for t in transpairs]
        inputnode.inputs.movingimgid = [t[1].get_imgid() for t in transpairs]
        imgidpairs = zip(inputnode.inputs.fixedimgid, inputnode.inputs.movingimgid)
        inputnode.inputs.transid = ['%s-%s'% (t[0], t[1]) for t in imgidpairs]

        trans_datasource = pe.MapNode(interface=nio.DataGrabber(
                                             infields=['fixedimgid', 'movingimgid'],
                                             outfields=['fixed_file', 'moving_file']),
                                      name='trans_datasource', 
                                      iterfield = ['fixedimgid', 'movingimgid'])
        trans_datasource.inputs.base_directory = os.path.abspath(join(self.dbpath, 'results'))
        #print (trans_datasource.inputs.base_directory)
        trans_datasource.inputs.template = 'preprocessed/_imgid_%s/norm_deformed.nii.gz'
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
        datasink.inputs.base_directory = os.path.abspath(join(self.dbpath, 'results', 'transformed'))
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
        wf.run(plugin='MultiProc', plugin_args={'n_procs' : multiprocessing.cpu_count()})# Compare two different transforms


    def transdiff(self, diffpairs):
        inputnode = pe.MapNode(interface=niu.IdentityInterface(fields=['sbj1a_imageid',
                                                                       'sbj1b_imageid',
                                                                       'sbj2a_imageid',
                                                                       'sbj2b_imageid']),
                               name='inputnode',
                               iterfield = ['sbj1a_imageid', 'sbj1b_imageid', 'sbj2a_imageid', 'sbj2b_imageid'])

        inputnode.inputs.sbj1a_imageid = [ diff[0][0].get_imgid() for diff in diffpairs ]
        inputnode.inputs.sbj1b_imageid = [ diff[0][1].get_imgid() for diff in diffpairs ]
        inputnode.inputs.sbj2a_imageid = [ diff[1][0].get_imgid() for diff in diffpairs ]
        inputnode.inputs.sbj2b_imageid = [ diff[1][1].get_imgid() for diff in diffpairs ]

        trans_datasource = pe.MapNode(interface=nio.DataGrabber(
                                             infields=['sbj1a_imageid', 'sbj1b_imageid', 'sbj2a_imageid', 'sbj2b_imageid'],
                                             outfields=['sbj1a_image', 'sbj1b_image', 'sbj2a_image', 'transform_2ab']),
                                      name='compose_datasource', 
                                      iterfield = ['sbj1a_imageid', 'sbj1b_imageid', 'sbj2a_imageid', 'sbj2b_imageid'])
        trans_datasource.inputs.base_directory = os.path.abspath(join(self.dbpath, 'results'))
        trans_datasource.inputs.template = '*'
        trans_datasource.inputs.field_template = dict(sbj1a_image   = join('preprocessed','_imgid_%s','deformed.nii.gz'),
                                                      sbj1b_image   = join('preprocessed','_imgid_%s','deformed.nii.gz'),
                                                      sbj2a_image   = join('preprocessed','_imgid_%s','deformed.nii.gz'),
                                                      transform_2ab = join('transformed','%s-%s','SyNQuick', 'transid*', 'out1Warp.nii.gz')                                            )
        trans_datasource.inputs.template_args = dict(sbj1a_image   = [['sbj1a_imageid']],
                                                     sbj1b_image   = [['sbj1b_imageid']],
                                                     sbj2a_image   = [['sbj2a_imageid']],
                                                     transform_2ab = [['sbj2a_imageid','sbj2b_imageid'],])
        trans_datasource.inputs.sort_filelist = True

        transnode = pe.MapNode(interface=SynQuick(),
                           name='transnode',
                           iterfield=['fixed_image', 'moving_image'])
        transnode.inputs.output_prefix = 'out'
        transnode.inputs.dimension = 3 

        composenode = pe.MapNode(interface=ComposeMultiTransform(), name='compose',
                   iterfield=['reference', 'transform1', 'transform2'])
        composenode.inputs.dimension = 3
        composenode.inputs.output_file = 'a2a.nii.gz'
        composenode.inputs.ignore_exception = False 

        resamplenode = pe.MapNode(interface=ApplyTransforms(), name='resample', 
                                  iterfield=['input_image', 'reference_image', 'transforms']) 
        resamplenode.inputs.dimension = 3
        resamplenode.inputs.output_image = 'resampled.nii'
        resamplenode.inputs.interpolation = 'Linear'
        resamplenode.inputs.default_value = 0
        resamplenode.inputs.invert_transform_flags = [False]

        similaritynode = pe.MapNode(interface=Similarity(), name='similarity', 
                                    iterfield=['volume1', 'volume2'])
        similaritynode.inputs.metric = 'cr'
        similaritynode.ignore_exception = False 

        outputnode = pe.Node(interface=niu.IdentityInterface(fields=['similarity']),
                                name='transdiff_outputnode')

        # build the workflow
        wf = pe.Workflow(name="transform")
        wf.connect([(inputnode, trans_datasource, [('sbj1a_imageid', 'sbj1a_imageid'),
                                                   ('sbj1b_imageid', 'sbj1b_imageid'),
                                                   ('sbj2a_imageid', 'sbj2a_imageid'),
                                                   ('sbj2b_imageid', 'sbj2b_imageid')]),
                    (trans_datasource, transnode, [('sbj2a_image', 'fixed_image'),
                                                   ('sbj1a_image', 'moving_image')]),
                    (trans_datasource, composenode, [('transform_2ab', 'transform1'),
                                                     ('sbj1b_image', 'reference')]),
                    (transnode, composenode, [('warp_field', 'transform2')]),
                    (composenode, resamplenode, [('output_file', 'transforms')]),
                    (trans_datasource, resamplenode, [('sbj1a_image', 'input_image')]),
                    (trans_datasource, resamplenode, [('sbj1b_image', 'reference_image')]),
                    (resamplenode, similaritynode, [('output_image', 'volume1')]),
                    (trans_datasource, similaritynode, [('sbj1b_image', 'volume2')]),
                    (similaritynode, outputnode, [('similarity', 'similarity')]),
                    ])
        # Run workflow with all cpus available
        g = wf.run(plugin='MultiProc', plugin_args={'n_procs' : multiprocessing.cpu_count()})
        return g 
