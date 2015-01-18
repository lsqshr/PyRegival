import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
from nipype.algorithms.metrics import Similarity
from nipype.interfaces.ants import ApplyTransforms
from additional_nipype_interfaces import *
from os.path import *
import multiprocessing
import itertools

# Compare two different transforms
def transdiff(dbpath, diffpairs):

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
    trans_datasource.inputs.base_directory = os.path.abspath(join(dbpath, 'results'))
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
    
def templateweighting():
    pass

def templatemerge():
    pass