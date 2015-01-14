import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
from nipype.algorithms.metrics import Similarity
from nipype.interfaces.ants import ApplyTransforms
from additional_nipype_interfaces import *
from os.path import *
import multiprocessing

# Compare two different transforms
def transdiff():
	inputnode = pe.MapNode(interface=niu.IdentityInterface(fields=['sbj1a_imageid',
	                                                               'sbj1b_imageid',
		                                                           'sbj2a_imageid',
		                                                           'sbj2b_imageid']),
                           name='inputnode',
                           iterfield = ['sbj1a_imageid', 'sbj1b_imageid', 'sbj2a_imageid', 'sbj2b_imageid'])
    trans_datasource = pe.MapNode(interface=nio.DataGrabber(
                                         infields=['sbj1a_imageid', 'sbj1b_imageid', 'sbj2a_imageid', 'sbj2b_imageid'],
                                         outfields=['sbj1a_image', 'sbj1b_image', 'sbj2a_image', 'transform_2ab']),
                                  name='compose_datasource', 
                                  iterfield = ['sbj1a_imageid', 'sbj1b_imageid', 'sbj2a_imageid', 'sbj2b_imageid'])
    trans_datasource.inputs.base_directory = os.path.abspath(join(dbpath, 'results'))
    trans_datasource.inputs.template = dict(sbj1a_image=join('preprocessed','_imgid_%s','ants_deformed.nii.gz'),
    	                                    sbj1b_image=join('preprocessed','_imgid_%s','ants_deformed.nii.gz'),
    	                                    sbj2a_image=join('preprocessed','_imgid_%s','ants_deformed.nii.gz'),
    	                                    transform_2ab=join('transformed','%s-%s','SynQuick', 'transid*', 'out1Warp.nii.gz')
    	                                    )
    trans_datasource.inputs.template_args = dict(sbj1a_image=[['sbj1a_imageid']],
	    	                                     sbj1b_image=[['sbj1b_imageid']],
	    	                                     sbj2a_image=[['sbj2a_imageid']] 
	    	                                     transform_2ab=[['sbj2a_imageid'],['sbj2b_imageid']])
	    	                                     )
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

    resamplenode = pe.MapNode(interface=ApplyTransforms(), name='resample', 
    	                      iterfield=['input_image', 'reference_image', 'transforms']) 
    resamplenode.inputs.dimension = 3
    resamplenode.inputs.output_image = 'resampled.nii'
    resamplenode.inputs.interpolation = 'Linear'
    resamplenode.inputs.default_value = 0
    resamplenode.inputs.invert_transform_flags = [False, False]

    similaritynode = pe.MapNode(interface=Similarity(), name='similarity', iterfield=['volume1', 'volume2'])
    similaritynode.inputs.metric = 'cr'
    similaritynode.ignore_exception = True

    # build the workflow
    wf = pe.Workflow(name="transform")
    wf.connect([(inputnode, trans_datasource, [('sbj1a_imageid', 'sbj1a_imageid'),
                                               ('sbj1b_imageid', 'sbj1b_imageid'),
                                               ('sbj2a_imageid', 'sbj2a_imageid'),
                                               ('sbj2b_imageid', 'sbj2b_imageid')]),
                (trans_datasource, transnode, [('sbj2a_image', 'fixed_image'),
                	                           ('sbj1a_image', 'moving_image')]),
                (transnode, composenode, [('warp_field', 'transform1')]),
                (trans_datasource, composenode, [('transform_2ab', 'transform2'),
                	                             ('sbj1b_image', 'reference')]),
                (composenode, resamplenode, [('output_file', 'transforms')]),
                (trans_datasource, resamplenode, [('sbj1a_image', 'input_image')]),
                (trans_datasource, resamplenode, [('sbj1b_image', 'reference_image')]),
                (resamplenode, similaritynode, [('output_image', 'volume1')]),
                (trans_datasource, similaritynode, [('sbj1b_image', 'volume2')])
		    	])
    return wf
    
def templateweighting():
	pass

def templatemerge():
	pass