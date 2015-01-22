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
from models import *
from additional_nipype_interfaces import *
import itertools
import pickle
import numpy as np

class MrRegival (object): 

    def __init__(self, dbpath):
        self.dbpath = dbpath
        self._collection = AdniMrCollection(dbpath=dbpath)
        self._ptemplate = None

    def build(self, lmodel = None, normtemplatepath='/usr/share/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz', 
                       normalise_method='ANTS',
                       interval=[12]):
        '''
        lmodel: a list of adnimrimg objects
        normtemplatepath: the template used by the normalisation

        '''
        if lmodel == None:
            lmodel = self._collection.getmrlist()

        self._collection.filtermodels(interval) # Only keep the usable cases
        self.normalise(normtemplatepath, normalise_method)
        transpairs = AdniMrCollection(lmodel).find_transform_pairs(interval)
        self.transform(transpairs)
        diffs = list(itertools.product(transpairs, repeat=2))
        # Make diffs to a {fid1,mid1,fid2,mid2: (interval1, interval2)} dict
        # Structure of diff itself: ((fid1,mid1, interval1), (fid2,mid2, interval2))
        g = self.transdiff(diffs) # transdiff does not consider the different interval

        for node in g.nodes():
            if node.name == 'similarity':
                similarity = node.result.outputs.similarity # The order is the same witht self.diffs

        self._ptemplate = {}
        self._ptemplate['transpairs'] = transpairs
        self._ptemplate['corr'] = dict(zip(diffs, similarity)) 
        self._ptemplate['lmodel'] = lmodel 

        with open(join(self.dbpath, 'ptemplate.pkl'), 'wb') as outfile:
            pickle.dump(self.pred_template, outfile)

        return self.pred_template


    def getcollection(self):
        return self._collection

    def load_ptemplate(self, filepath='ptemplate.pkl'):
        with open(filepath, 'rb') as infile:
            self._ptemplate = pickle.load(infile)
        return self._ptemplate


    def normalise(self, normtemplatepath='/usr/share/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz', 
                normalise_method='ANTS', lmodel=None):
        ''' 

        '''
        if lmodel == None:
            lmodel = self._collection.getmrlist()              
        
        lsbjid = [model.getmetafield('Subject') for model in lmodel]
        limgid = [model.getimgid() for model in lmodel]
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


    def transform(self, transpairs, lmodel=None):

        if lmodel == None:
            lmodel = self._collection.getmrlist()

        # Make the workflow
        inputnode = pe.MapNode(interface=niu.IdentityInterface(fields=['fixedimgid', 'movingimgid', 'transid']),
                               name='inputnode',
                               iterfield = ['fixedimgid', 'movingimgid', 'transid'])
        inputnode.inputs.fixedimgid  = [t.fixedimage.getimgid() for t in transpairs]
        inputnode.inputs.movingimgid = [t.movingimage.getimgid() for t in transpairs]
        imgidpairs = zip(inputnode.inputs.movingimgid, inputnode.inputs.fixedimgid)
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
        inputnode = pe.MapNode(interface=niu.IdentityInterface(fields=['sbj1_mov_imgid',
                                                                       'sbj1_fix_imgid',
                                                                       'sbj2_mov_imgid',
                                                                       'sbj2_fix_imgid']),
                               name='inputnode',
                               iterfield = ['sbj1_mov_imgid', 'sbj1_fix_imgid', 'sbj2_mov_imgid', 'sbj2_fix_imgid'])

        inputnode.inputs.sbj1_mov_imgid = [ diff[0].movingimage.getimgid() for diff in diffpairs ]
        inputnode.inputs.sbj1_fix_imgid = [ diff[0].fixedimage.getimgid() for diff in diffpairs ]
        inputnode.inputs.sbj2_mov_imgid = [ diff[1].movingimage.getimgid() for diff in diffpairs ]
        inputnode.inputs.sbj2_fix_imgid = [ diff[1].fixedimage.getimgid() for diff in diffpairs ]

        trans_datasource = pe.MapNode(interface=nio.DataGrabber(
                                             infields=['sbj1_mov_imgid', 'sbj1_fix_imgid', 'sbj2_mov_imgid', 'sbj2_fix_imgid'],
                                             outfields=['sbj1_mov_img', 'sbj1_fix_img', 'sbj2_mov_img', 'transform_2ab']),
                                      name='compose_datasource', 
                                      iterfield = ['sbj1_mov_imgid', 'sbj1_fix_imgid', 'sbj2_mov_imgid', 'sbj2_fix_imgid'])
        trans_datasource.inputs.base_directory = os.path.abspath(join(self.dbpath, 'results'))
        trans_datasource.inputs.template = '*'
        trans_datasource.inputs.field_template = dict(sbj1_mov_img   = join('preprocessed','_imgid_%s','norm_deformed.nii.gz'),
                                                      sbj1_fix_img   = join('preprocessed','_imgid_%s','norm_deformed.nii.gz'),
                                                      sbj2_mov_img   = join('preprocessed','_imgid_%s','norm_deformed.nii.gz'),
                                                      transform_2ab = join('transformed','%s-%s','SyNQuick', 'transid*', 'out1Warp.nii.gz')                                            )
        trans_datasource.inputs.template_args = dict(sbj1_mov_img   = [['sbj1_mov_imgid']],
                                                     sbj1_fix_img   = [['sbj1_fix_imgid']],
                                                     sbj2_mov_img   = [['sbj2_mov_imgid']],
                                                     transform_2ab = [['sbj2_mov_imgid','sbj2_fix_imgid'],])
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
        wf.connect([(inputnode, trans_datasource, [('sbj1_mov_imgid', 'sbj1_mov_imgid'),
                                                   ('sbj1_fix_imgid', 'sbj1_fix_imgid'),
                                                   ('sbj2_mov_imgid', 'sbj2_mov_imgid'),
                                                   ('sbj2_fix_imgid', 'sbj2_fix_imgid')]),
                    (trans_datasource, transnode, [('sbj2_mov_img', 'fixed_image'),
                                                   ('sbj1_mov_img', 'moving_image')]),
                    (trans_datasource, composenode, [('transform_2ab', 'transform1'),
                                                     ('sbj1_fix_img', 'reference')]),
                    (transnode, composenode, [('warp_field', 'transform2')]),
                    (composenode, resamplenode, [('output_file', 'transforms')]),
                    (trans_datasource, resamplenode, [('sbj1_mov_img', 'input_image')]),
                    (trans_datasource, resamplenode, [('sbj1_fix_img', 'reference_image')]),
                    (resamplenode, similaritynode, [('output_image', 'volume1')]),
                    (trans_datasource, similaritynode, [('sbj1_fix_img', 'volume2')]),
                    (similaritynode, outputnode, [('similarity', 'similarity')]),
                    ])

        # Run workflow with all cpus available
        g = wf.run(plugin='MultiProc', plugin_args={'n_procs' : multiprocessing.cpu_count()})
        return g


    def predict(self, targetpair, N=0, t=0.5, rowweight=0.5, colweight=0.5, real_followup=None, option='change', ptemplate=None):
        '''
        targetpair: tuple (mrid1, mrid2, interval)
        N : Int the number of neighbours to merge 
        t : Gaussian kernel density
        rowweight: the relative weight of the row elements for neighbood building
        colweight: the relative weight of the column elements for neighbood building
        option: 'change'/'baseline image'/'both'
        '''

        if ptemplate == None:
            ptemplate = self._ptemplate

        # Convert the similarity dict to a matrix with the order of the mrid pairs
        simmatrix, elligible_pairs = self._convert_ptemplate2matrix(interval=targetpair.getinterval()) 
        # TODO: If this subject is not in the template, add this subject to the template

        # Find the column and row of this subject, 
        print targetpair.fixedimage.getimgid()
        print [ p.movingimage.getimgid() + '-' + p.fixedimage.getimgid() for p in elligible_pairs]
        print [ p.movingimage.getimgid() + '-' + p.fixedimage.getimgid() for p in self._ptemplate['transpairs']]
        targetidx = elligible_pairs.index(targetpair)
        matrow = simmat[targetidx, :]
        matcol = simmat[:, targetidx]

        # Assign itself with similairity of 0 in the matrix to ignore it when merge
        allrid = [p.fixedimage.getmetafield['RID'] for p in elligible_pairs]
        targetrid = targetpair.fixedimage.getmetafield('RID')
        all_target_sbj_idx = [i for i, x in enumerate(allrid) if x == targetrid]
        matcol[targetidx] = matrow[all_target_sbj_idx] = 0

        #linterval = np.array([p.fixed_image.getviscode() - p.moving_image.getviscode()
        #                      for p in elligible_pairs])

        #np.abs(linterval - targetpair[2])
        # Calculate the row&column weights distribution considering the interval
        dcol = 1 - matrow
        drow = 1 - drow
        ecol = exp(-(dcol/t))
        erow = exp(-(drow/t))
        w = (colweight * ecol + rowweight * erow) / (colweight + rowweight)

        # Ignore for now : Find the top N neighbours from row/column by weighting 

        # Find the followup pairs of templates for merging
        followuppairs = self._collection.find_followups(elligible_pairs, interval) 

        # Merge these templates by weighting
        g = self.merge(followingpairs, w)
        

    def _convert_ptemplate2matrix(self, interval=None):
        pairs = self._ptemplate['transpairs']
        sim = self._ptemplate['corr']

        # Remove the transpairs without following transpairs
        elligible_pairs = self._collection.filter_elligible_pairs(pairs, interval)
        simmat = np.zeros((len(elligible_pairs), len(elligible_pairs)))

        for i, p1 in enumerate(elligible_pairs):
            for j, p2 in enumerate(elligible_pairs):
                simmat[i,j] = sim[(p1,p2)][0]

        return simmat, elligible_pairs


    def merge(self, pairs, w, targetpair, followup_id = None):
        inputnode = pe.MapNode(interface=niu.IdentityInterface(fields=['transa_imageid',
                                                                       'transb_imageid',
                                                                       'targetb_imageid']),
                               name='inputnode',
                               iterfield = ['transa_imageid', 'transb_imageid'])

        inputnode.inputs.transa_imageid = [ pair.movingimage.getimgid() for pair in pairs ]
        inputnode.inputs.transb_imageid = [ pair.fixedimage.getimgid() for pair in pairs ]
        inputnode.inputs.targeta_imageid = targetpair.movingimage.getimgid()
        inputnode.inputs.targetb_imageid = targetpair.fixedimage.getimgid()

        # Grab the transforms by id
        pred_datasource = pe.MapNode(interface=nio.DataGrabber(
                                             infields=['transa_imageid', 'transb_imageid', 'targetb_imageid'],
                                             outfields=['transa_image', 'targetb_image', 'trans']),
                                      name='compose_datasource', 
                                      iterfield = ['transa_imageid', 'transb_imageid'])
        pred_datasource.inputs.base_directory = os.path.abspath(join(self.dbpath, 'results'))
        pred_datasource.inputs.template = '*'
        pred_datasource.inputs.field_template = dict(transa_image  = join('preprocessed','_imgid_%s','deformed.nii.gz'),
                                                      targetb_image = join('preprocessed','_imgid_%s','deformed.nii.gz'),
                                                      trans = join('transformed','%s-%s','SyNQuick', 'transid*', 'out1Warp.nii.gz'))
        pred_datasource.inputs.template_args = dict(transa_image   = [['transa_imageid']],
                                                     targetb_image   = [['targetb_imageid']],
                                                     trans = [['transa_imageid','transb_imageid']])
        pred_datasource.inputs.sort_filelist = True

        # Transform each IA to the target IA
        transnode = pe.MapNode(interface=SynQuick(),
                   name='transnode',
                   iterfield=['fixed_image', 'moving_image'])
        transnode.inputs.output_prefix = 'out'
        transnode.inputs.dimension = 3

        # Compose each transform with the space matching
        composenode = pe.MapNode(interface=ComposeMultiTransform(), name='compose',
                   iterfield=['reference', 'transform1', 'transform2'])
        composenode.inputs.dimension = 3
        composenode.inputs.output_file = 'a2a.nii.gz'

        # Weight Sum the transforms
        weighted_sum_node = pe.Node(interface=WeightedSumTrans(), name='weighted_sum')
        weighted_sum_node.inputs.weights = w

        # warp the target second image
        resamplenode = pe.MapNode(interface=ApplyTransforms(), name='resample', 
                                  iterfield=['input_image', 'reference_image', 'transforms']) 
        resamplenode.inputs.dimension = 3
        resamplenode.inputs.output_image = 'resampled.nii'
        resamplenode.inputs.interpolation = 'Linear'
        resamplenode.inputs.default_value = 0
        resamplenode.inputs.invert_transform_flags = [False]

        # Evalutate the similarity of the predicted image
        similaritynode = pe.Node(interface=Similarity(), name='similarity')
        similaritynode.inputs.metric = 'cr'

        # Data Sink to save the summed transform and the resampled image
        datasink = pe.Node(nio.DataSink(infields=['container',
                                                  'predicted', 
                                                  'predicted.@trans_image',
                                                  'SyNQuick.@predicted_trans']))
        datasink.inputs.base_directory = os.path.abspath(join(self.dbpath, 'results', 'predicted'))
        #datasink.inputs.substitutions = [('_transnode', 'transid')]
        datasink.inputs.parameterization = True

        outputnode = pe.Node(interface=niu.IdentityInterface(fields=['similarity']), name='evalsimilarity')

        wf = pe.Workflow(name='prediction')
        wf.connect([(inputnode, pred_datasource, [('transa_imageid', 'transa_imageid'), 
                                                   ('transb_imageid', 'transb_imageid'), 
                                                   ('targetb_imageid','targetb_imageid'),
                                                  ]),
                    (pred_datasource, transnode, [('transa_image', 'moving_image'),
                                                   ('targetb_image', 'fixed_image')]),
                    (pred_datasource, composenode, [('trans', 'transform1'),
                                                     ('targetb_image', 'reference')]),
                    (transnode, composenode, [('warpfield', 'transform2')]),
                    (composenode, weighted_sum_node, [('output_file', 'transforms')]),
                    (weighted_sum_node, resamplenode, [('out_trans','transforms')]),
                    (pred_datasource, resamplenode, [('targetb_image','input_image')]),
                    (resamplenode, datasink, [('output_image','predicted.@trans_image')]),
                    (transnode, datasink, [('trans', 'predicted.@predicted_trans')])
                   ])
        if followup_id != None:
            # Grab the follow up image
            followup_datasource = pe.Node(interface=nio.DataGrabber(infields=['followupid'],
                                                                    outfields=['followupimage']),
                                                                    name='followup_datasource')
            followup_datasource.inputs.base_directory = os.path.abspath(join(self.dbpath, 'results'))
            followup_datasource.inputs.template = '*'
            followup_datasource.inputs.field_template = dict(followupimage = join('preprocessed','_imgid_%s','deformed.nii.gz'))
            followup_datasource.inputs.template_args = dict(followupimage = [['followupid']])
            followup_datasource.inputs.sort_filelist = True

            wf.connect(followup_datasource, 'followupimage', similaritynode, 'Volume1')
            wf.connect(resamplenode, 'output_image', similaritynode, 'Volume2')
            wf.connect(similarity, 'similarity', outputnode, 'similarity')

        g = wf.run(plugin='MultiProc', plugin_args={'n_procs' : multiprocessing.cpu_count()})
        return g