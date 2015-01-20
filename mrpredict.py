from utils import *
import numpy as np
from nipype.interfaces.ants import Registration
from nipype.interfaces.ants import ApplyTransforms
from nipype.algorithms.metrics import Similarity
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
from additional_nipype_interfaces import *

class MrPredictor(object):
    def __init__(self, dbpath, ptemplate):
        self.dbpath = dbpath
        self.ptemplate = ptemplate
        self.collection = AdniMrCollection(self.ptemplate['lmodel'])

    def predict(self, targetpair, N=0, t=0.5, rowweight=0.5, colweight=0.5, real_followup=None, option='change'):
        '''
        targetpair: tuple (mrid1, mrid2, interval)
        N : Int the number of neighbours to merge 
        t : Gaussian kernel density
        rowweight: the relative weight of the row elements for neighbood building
        colweight: the relative weight of the column elements for neighbood building
        option: 'change'/'baseline image'/'both'
        '''

        # Convert the similarity dict to a matrix with the order of the mrid pairs
        simmatrix, elligible_pairs = self._convert_ptemplate2matrix(interval=targetpair[2]) 
        print elligible_pairs
        # TODO: If this subject is not in the template, add this subject to the template

        # Find the column and row of this subject, 
        targetidx = elligible_pairs.index(targetpair)
        matrow = simmat[targetidx, :]
        matcol = simmat[:, targetidx]
        linterval = np.array([p[2] for p in elligible_pairs])
        np.abs(linterval - targetpair[2])

        # calculate the row&column weights distribution considering the interval
        dcol = 1 - matrow
        drow = 1 - drow
        ecol = exp(-(dcol/t))
        erow = exp(-(drow/t))
        w = (colweight * ecol + rowweight * erow) / (colweight + rowweight)

        # Ignore for now : Find the top N neighbours from row/column by weighting 

        # Find the followup pairs of templates for merging
        followuppairs = self.collection.find_followups(elligible_pairs, interval) 

        # Merge these templates by weighting
        g = self.trans_weighted_merge(followingpairs, w)
        

    def _convert_ptemplate2matrix(self, interval=None):
        pairs = self.ptemplate['transpairs']
        sim = self.ptemplate['corr']

        # Remove the transpairs without following transpairs
        elligible_pairs = self.collection.filter_elligible_pairs(pairs, interval)
        simmat = np.zeros((len(elligible_pairs), len(elligible_pairs)))

        for i, p1 in enumerate(elligible_pairs):
            for j, p2 in enumerate(elligible_pairs):
                simmat[i,j] = sim[(p1,p2)][0]

        return simmat, elligible_pairs


    def trans_weighted_merge(self, pairs, w, targetpair, followup_id = None):
        inputnode = pe.MapNode(interface=niu.IdentityInterface(fields=['transa_imageid',
                                                                       'transb_imageid',
                                                                       'targetb_imageid']),
                               name='inputnode',
                               iterfield = ['transa_imageid', 'transb_imageid'])

        inputnode.inputs.transa_imageid = [ pair[1] for pair in pairs ]
        inputnode.inputs.transb_imageid = [ pair[0] for pair in pairs ]
        inputnode.inputs.targeta_imageid = targetpair[1]
        inputnode.inputs.targetb_imageid = targetpair[0]

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