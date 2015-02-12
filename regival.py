from nipype.interfaces import fsl
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
from nipype.interfaces.ants.legacy import antsIntroduction
from nipype.interfaces.ants import Registration
from nipype.interfaces.ants import ApplyTransforms 
from nipype.interfaces.utility import Function
from nipype.algorithms.metrics import ErrorMap
from os.path import * 
import os
import multiprocessing
from models import *
from additional_nipype_interfaces import *
import itertools
import pickle
import numpy as np
import time

class MrRegival (object): 

    def __init__(self, dbpath=None, collection=None):
        self.dbpath = dbpath
        assert not(dbpath is None and collection is None) 

        if collection is not None:
            self._collection = collection
        elif dbpath is not None: # Extract new collection when collection is not given
            if exists(join(dbpath, 'dbgen.csv')):
                self._collection = AdniMrCollection(dbpath=dbpath, regendb=False)
            else:
                self._collection = AdniMrCollection(dbpath=dbpath, regendb=True)

        self._ptemplate = None
        self._log = []


    def getcollection(self):
        return self._collection


    def setcollection(self, collection):
        self._collection = collection


    def load_ptemplate(self, filepath='ptemplate.pkl'):
        with open(filepath, 'rb') as infile:
            self._ptemplate = pickle.load(infile)
        return self._ptemplate


    def normalise(self, normtemplatepath='MNI152_T1_2mm_brain.nii.gz', 
                normalise_method='FSL', lmodel=None, ignoreexception=False, ncore=2):
        ''' 
        Normalisation Pipeline with either FSL flirt or ANTS antsIntroduction

        '''
        normtemplatepath = abspath(normtemplatepath)
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
        stdroi.inputs.ignore_exception = ignoreexception
        stripper = pe.Node(fsl.BET(), name='stripper')
        stripper.inputs.frac = 0.25
        stripper.inputs.ignore_exception = ignoreexception

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
            normwarp.inputs.ignore_exception = ignoreexception
            infield  = 'in_file'
            outfield = 'out_file'
        elif normalise_method == 'ANTS':
            normwarp = pe.Node(antsIntroduction(), name='ants')
            normwarp.inputs.reference_image = normtemplatepath
            normwarp.inputs.max_iterations = [30,90,20]
            normwarp.inputs.num_threads = 1 # This parameter will not take effects
            normwarp.inputs.transformation_model = 'RI'
            normwarp.inputs.out_prefix = 'norm_'
            normwarp.inputs.ignore_exception = ignoreexception
            infield  = 'input_image'
            outfield = 'output_file'

        # The input/output file spec were differnet between FSL and ANTS in nipype
        wf.connect(stripper, 'out_file', normwarp, infield)
        #wf.connect(inputidnode, 'subject_id', datasink, 'container')
        wf.connect(normwarp, outfield, datasink, 'preprocessed')

        # Run workflow with all cpus available
        wf.run(plugin='MultiProc', plugin_args={'n_procs' : ncore})


    def transform(self, transpairs=None, ignoreexception=False, ncore=2):

        if transpairs == None:
            self._collection.filtermodels() # filter the models when this method is called externally 
            transpairs = self._collection.find_transform_pairs()

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
        trans_datasource.inputs.template = 'preprocessed/_imgid_%s/norm_deformed.nii.gz'
        trans_datasource.inputs.template_args = dict(fixed_file=[['fixedimgid']],
                                                     moving_file=[['movingimgid']])
        trans_datasource.inputs.sort_filelist = True

        transnode = pe.MapNode(interface=SynQuick(),
                               name='transnode',
                               iterfield=['fixed_image', 'moving_image'])
        transnode.inputs.output_prefix = 'out'
        transnode.inputs.dimension = 3 
        transnode.inputs.ignore_exception = ignoreexception
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
        wf.run(plugin='MultiProc', plugin_args={'n_procs' : ncore})# Compare two different transforms


    def transdiff(self, diffpairs=None, option='trans', ignoreexception=False, ncore=2):
        '''
        diffpairs: list of (pair1, pair2)
        option: 'trans'/'image'
        '''
        if diffpairs == None:
            diffpairs = self._collection.getdiffpairs()
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
                                             outfields=['sbj1_mov_img', 'sbj1_fix_img', 'sbj2_mov_img', 'sbj2_fix_img', 'transform_2ab']),
                                      name='compose_datasource', 
                                      iterfield = ['sbj1_mov_imgid', 'sbj1_fix_imgid', 'sbj2_mov_imgid', 'sbj2_fix_imgid'])
        trans_datasource.inputs.base_directory = os.path.abspath(join(self.dbpath, 'results'))
        trans_datasource.inputs.template = '*'
        trans_datasource.inputs.field_template = dict(sbj1_mov_img   = join('preprocessed','_imgid_%s','norm_deformed.nii.gz'),
                                                      sbj1_fix_img   = join('preprocessed','_imgid_%s','norm_deformed.nii.gz'),
                                                      sbj2_mov_img   = join('preprocessed','_imgid_%s','norm_deformed.nii.gz'),
                                                      sbj2_fix_img   = join('preprocessed','_imgid_%s','norm_deformed.nii.gz'),
                                                      transform_2ab = join('transformed','%s-%s','SyNQuick', 'transid*', 'out1Warp.nii.gz'))
        trans_datasource.inputs.template_args = dict(sbj1_mov_img   = [['sbj1_mov_imgid']],
                                                     sbj1_fix_img   = [['sbj1_fix_imgid']],
                                                     sbj2_mov_img   = [['sbj2_mov_imgid']],
                                                     sbj2_fix_img   = [['sbj2_fix_imgid']],
                                                     transform_2ab = [['sbj2_mov_imgid','sbj2_fix_imgid']])
        trans_datasource.inputs.sort_filelist = True

        transnode = pe.MapNode(interface=SynQuick(),
                           name='transnode',
                           iterfield=['fixed_image', 'moving_image'])
        transnode.inputs.output_prefix = 'out'
        transnode.inputs.dimension = 3 
        transnode.inputs.ignore_exception = ignoreexception

        composenode = pe.MapNode(interface=ComposeMultiTransform(), name='compose',
                   iterfield=['reference', 'transform1', 'transform2'])
        composenode.inputs.dimension = 3
        composenode.inputs.output_file = 'a2a.nii.gz'
        composenode.inputs.ignore_exception = ignoreexception

        resamplenode = pe.MapNode(interface=ApplyTransforms(), name='resample', 
                                  iterfield=['input_image', 'reference_image', 'transforms']) 
        resamplenode.inputs.dimension = 3
        resamplenode.inputs.output_image = 'resampled.nii'
        resamplenode.inputs.interpolation = 'Linear'
        resamplenode.inputs.default_value = 0
        resamplenode.inputs.invert_transform_flags = [False]
        resamplenode.inputs.ignore_exception = ignoreexception

        errmapnode = pe.MapNode(interface=ErrorMap(), name='errmap', 
                                    iterfield=['in_ref', 'in_tst'])
        errmapnode.ignore_exception = ignoreexception

        outputnode = pe.Node(interface=niu.IdentityInterface(fields=['distance']),
                                name='transdiff_outputnode')

        # build the workflow
        wf = pe.Workflow(name="transdiff")
        if option == 'trans':
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
                        (resamplenode, errmapnode, [('output_image', 'in_ref')]),
                        (trans_datasource, errmapnode, [('sbj1_fix_img', 'in_tst')]),
                        (errmapnode, outputnode, [('distance', 'distance')]),
                        ])
        else:
            wf.connect([(inputnode, trans_datasource, [('sbj1_mov_imgid', 'sbj1_mov_imgid'),
                                                       ('sbj1_fix_imgid', 'sbj1_fix_imgid'),
                                                       ('sbj2_mov_imgid', 'sbj2_mov_imgid'),
                                                       ('sbj2_fix_imgid', 'sbj2_fix_imgid')]),
                        (trans_datasource, errmapnode, [('sbj2_fix_img', 'in_ref'),
                                                        ('sbj1_fix_img', 'in_tst')]),
                        (errmapnode, outputnode, [('distance', 'distance')]),
                        ])

        # Run workflow with all cpus available
        g = wf.run(plugin='MultiProc', plugin_args={'n_procs' : ncore})
        for node in g.nodes():
            if node.name == 'errmap':
                return node.result.outputs.distance # The order is the same with self.diffs


    def predict(self, targetpair, tpairs, w, K=4, t=0.5, decayratio=0.85, real_followupid=None, ncore=2, outprefix=''):
        '''
        targetpair : the pair to be simulated 
        tpairs : template pairs that were used for comparison 
        w : weights of the template; this weights can be totally coustmomised
        N : Int the number of neighbours to merge 
        t : Gaussian kernel density
        real_followupid: the followup image id for evaluation
        ncore : n cpu cores to run the workflow
        '''
        start = time.time()
        #print 'raw distances are: ', w
        w = w - np.average(w)
        w = w / np.max(np.abs(w))
        ew = np.exp(-(w/t))
        # Find the top N neighbours from row/column by weighting 
        ewcopy = np.array(ew, copy=True)
        sortidx = np.argsort(ewcopy)
        #ewcopy.sort()
        selectedidx = sortidx[-K:]

        #cutthreshold = ewcopy[-K]
        #condition =  ew >= cutthreshold
        ew = ew[selectedidx] 
        from operator import itemgetter 
        tpairs = itemgetter(*selectedidx)(tpairs)

        # Find the followup pairs of templates for merging
        followuppairs = self._collection.find_followups(tpairs, interval=[targetpair.getinterval()]) 

        # Merge these templates by weighting
        g = self.merge(followuppairs, ew, targetpair, decayratio=decayratio, ncore=ncore, outprefix=outprefix)

        distance = None
        if real_followupid is not None:
            for node in g.nodes():
                if node.name == 'errmap':
                    distance = node.result.outputs.distance

            print 'The resampling distance is: ', distance 
        else:
            'The real followup id is not provided, no comparison was possible...'

        end = time.time()
        logstr = 'The prediction of 1 target took %f seconds, %f in %d elligible pairs' % \
                  ((end-start), (end-start)/len(self._collection.filter_elligible_pairs()), len(self._collection.filter_elligible_pairs()))
        print logstr
        self._log.append(logstr)

        return distance


    def merge(self, pairs, w, targetpair, decayratio=0.85, ncore=2, outprefix=''):
        inputnode = pe.MapNode(interface=niu.IdentityInterface(fields=['transa_imageid',
                                                                       'transb_imageid',
                                                                       'targetb_imageid',
                                                                       'real_followupid',
                                                                       'w', 
                                                                       'transid',
                                                                       'decayratio']),
                               name='inputnode',
                               iterfield = ['transa_imageid', 'transb_imageid'])

        inputnode.inputs.transa_imageid = [ pair.movingimage.getimgid() for pair in pairs ]
        inputnode.inputs.transb_imageid = [ pair.fixedimage.getimgid() for pair in pairs ]
        inputnode.inputs.targeta_imageid = targetpair.movingimage.getimgid()
        inputnode.inputs.targetb_imageid = targetpair.fixedimage.getimgid()
        inputnode.inputs.real_followupid = self._collection.find_followups([targetpair], [targetpair.getinterval()])[0].fixedimage.getimgid()
        inputnode.inputs.decayratio = decayratio
        print inputnode.inputs.real_followupid
        assert inputnode.inputs.real_followupid != None
        inputnode.inputs.w = w
        inputnode.inputs.transid = '%s-%s' % (targetpair.movingimage.getimgid(), targetpair.fixedimage.getimgid())

        # Grab the transforms by id
        pred_datasource = pe.MapNode(interface=nio.DataGrabber(
                                             infields=['transa_imageid', 'transb_imageid', 'targetb_imageid'],
                                             outfields=['transa_image', 'targetb_image', 'trans']),
                                      name='pred_datasource', 
                                      iterfield = ['transa_imageid', 'transb_imageid'])
        pred_datasource.inputs.base_directory = os.path.abspath(join(self.dbpath, 'results'))
        pred_datasource.inputs.template = '*'
        pred_datasource.inputs.field_template = dict(transa_image  = join('preprocessed','_imgid_%s','norm_deformed.nii.gz'),
                                                      targetb_image = join('preprocessed','_imgid_%s','norm_deformed.nii.gz'),
                                                      trans = join('transformed','%s-%s','SyNQuick', 'transid*', 'out1Warp.nii.gz'))
        pred_datasource.inputs.template_args = dict(transa_image   = [['transa_imageid']],
                                                     targetb_image   = [['targetb_imageid']],
                                                     trans = [['transa_imageid','transb_imageid']])
        pred_datasource.inputs.sort_filelist = True

        # The target datasource is for the final resampling, it only requires one filepath
        target_datasource = pe.Node(interface=nio.DataGrabber(infields=['targetb_imageid', 'real_followupid'],
                                                              outfields=['targetb_image', 'real_followupimage']), 
                                                              name='target_datasource')
        target_datasource.inputs.base_directory = os.path.abspath(join(self.dbpath, 'results'))
        target_datasource.inputs.template = '*'
        target_datasource.inputs.field_template = dict(targetb_image=join('preprocessed','_imgid_%s','norm_deformed.nii.gz'),
                                                       real_followupimage=join('preprocessed','_imgid_%s','norm_deformed.nii.gz'))
        target_datasource.inputs.template_args = dict(targetb_image=[['targetb_imageid']], real_followupimage=[['real_followupid']])
        #target_datasource.inputs.targetb_imageid = [targetpair.fixedimage.getimgid()]
        target_datasource.inputs.sort_filelist = True

        # Transform each IA to the target IA
        transnode = pe.MapNode(interface=SynQuick(),
                   name='transnode',
                   iterfield=['fixed_image', 'moving_image'])
        transnode.inputs.output_prefix = 'out'
        transnode.inputs.dimension = 3
        transnode.ignore_exception = True

        # Compose each transform with the space matching
        composenode = pe.MapNode(interface=ComposeMultiTransform(), name='compose',
                   iterfield=['reference', 'transform1', 'transform2'])
        composenode.inputs.dimension = 3
        composenode.inputs.output_file = 'a2a.nii.gz'
        composenode.inputs.ignore_exception = True

        # Weight Sum the transforms
        weighted_sum_node = pe.Node(interface=Function(input_names=['transforms', 'weights', 'decayratio'],
                                                       output_names=['out_file'],
                                                       function=trans_weighted_sum), 
                                    name='weighted_sum')

        # warp the target second image
        resamplenode = pe.Node(interface=ApplyTransforms(), name='resample')  # refrence_image is set as the input_image itself now
        resamplenode.inputs.dimension = 3
        resamplenode.inputs.output_image = 'resampled.nii'
        resamplenode.inputs.interpolation = 'Linear'
        resamplenode.inputs.default_value = 0
        resamplenode.inputs.invert_transform_flags = [False]
        resamplenode.inputs.ignore_exception = True

        # Evalutate the distance of the predicted image
        errmapnode = pe.Node(interface=ErrorMap(), name='errmap')

        # Data Sink to save the summed transform and the resampled image
        datasink = pe.Node(nio.DataSink(infields=['container',
                                                  'predicted', 
                                                  'predicted.@trans_image',
                                                  'predicted.@predicted_trans',
                                                  'predicted.@errmap'
                                                  ]), name='mergesink')
        datasink.inputs.base_directory = os.path.abspath(join(self.dbpath, 'results', outprefix + 'predicted'))
        datasink.inputs.parameterization = True

        outputnode = pe.Node(interface=niu.IdentityInterface(fields=['distance']), name='evaldistance')

        wf = pe.Workflow(name='prediction')
        wf.connect([(inputnode, pred_datasource, [('transa_imageid', 'transa_imageid'), 
                                                   ('transb_imageid', 'transb_imageid'), 
                                                   ('targetb_imageid','targetb_imageid')
                                                  ]),
                    (inputnode, target_datasource, [('real_followupid', 'real_followupid'),
                                                    ('targetb_imageid', 'targetb_imageid')]),
                    (inputnode, weighted_sum_node, [('w', 'weights'),
                                                    ('decayratio', 'decayratio')]),
                    (pred_datasource, transnode, [('transa_image', 'moving_image'),
                                                   ('targetb_image', 'fixed_image')]),
                    (pred_datasource, composenode, [('trans', 'transform1'),
                                                     ('targetb_image', 'reference')]),
                    (transnode, composenode, [('warp_field', 'transform2')]),
                    (composenode, weighted_sum_node, [('output_file', 'transforms')]),
                    (weighted_sum_node, resamplenode, [('out_file','transforms')]),
                    (target_datasource, resamplenode, [('targetb_image','input_image')]),
                    (target_datasource, resamplenode, [('real_followupimage','reference_image')]),
                    (resamplenode, datasink, [('output_image','predicted.@trans_image')]),
                    (inputnode, datasink, [('transid', 'container')]),
                    (weighted_sum_node, datasink, [('out_file', 'predicted.@predicted_trans')])
                   ])
        # Grab the follow up image
        wf.connect(target_datasource, 'real_followupimage', errmapnode, 'in_ref')
        wf.connect(resamplenode, 'output_image', errmapnode, 'in_tst')
        wf.connect(errmapnode, 'distance', outputnode, 'distance')
        wf.connect(errmapnode, 'out_map', datasink, 'predicted.@errmap')

        g = wf.run(plugin='MultiProc', plugin_args={'n_procs' : ncore})
        return g


    def printlog(self):
        print '\n'.join(self._log)


def trans_weighted_sum(transforms, weights, decayratio):
    import nibabel as nib
    import numpy as np
    import os

    for i, (path, w) in enumerate(zip(transforms, weights)):
        img = nib.load(path)
        data = img.get_data()
        wdata = data * w

        if i == 0:
            merged = wdata
            affine = img.get_affine()
            header = img.header
        else:
            merged +=wdata

    merged = merged * decayratio/ np.sum(weights) # Normalise
    newtrans = nib.Nifti1Image(merged, affine)
    # Copy header, otherwise the transform will be shifted
    # Interstingly that nibabel won't allow you to assign the header directly
    for key in header:
        newtrans.header[key] = header[key]
    outfile = os.path.join(os.getcwd(), 'newtrans.nii.gz')
    nib.save(newtrans, outfile)

    return outfile 
