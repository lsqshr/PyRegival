import os
import os.path as op # Convension used in nipype.algorithms.metrics
from nipype.interfaces.fsl.preprocess import FSLCommandInputSpec, FSLCommand
from nipype.interfaces.ants.registration import ANTSCommand, ANTSCommandInputSpec, RegistrationOutputSpec
from nipype.interfaces.base import TraitedSpec, File, traits, CommandLine, InputMultiPath
from nipype.interfaces.traits_extension import isdefined
from nipype.algorithms.metrics import ErrorMap, ErrorMapOutputSpec
import nibabel as nb # Convension used in nipype.algorithms.metrics
import numpy as np # Convension used in nipype.algorithms.metrics

#------------------
# Wrap 'FSL/standard_space_roi' of FSL, did not find this interface in the current nipype
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


class StdRoi(FSLCommand):
    _cmd = 'standard_space_roi'
    input_spec = StdRoiInputSpec
    output_spec = StdRoiOutputSpec


#------------------
# Wrap 'ants/antsRegistrationSyNQuick'
class SynQuickInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='-d %d', usedefault=False,
                            position=1, desc='image dimension (2 or 3)')
    num_threads = traits.Int(usedefault=True, argstr='-n %d',
                             nohash=True, desc="Number of ITK threads to use")
    fixed_image = InputMultiPath(File(exists=True), mandatory=True, argstr='-f %s',
                                    desc='image to apply transformation to (generally a coregistered '
                                            'functional)')
    moving_image = InputMultiPath(File(exists=True), argstr='-m %s',
                                    mandatory=True,
                                    desc='image to apply transformation to (generally a coregistered '
                                            'functional)')
    output_prefix = traits.Str('out', usedefault=True,
                                            argstr='-o %s',
                                            mandatory=True, desc='')


class SynQuickOutputSpec(TraitedSpec):
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


## Wrap antsBrainExtraction.sh 
class antsBrainExtractionInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, mandatory=True, argstr='-d %d', usedefault=False,
                            desc='2 or 3 (for 2- or 3-dimensional image)')
    in_file = File(exists=True, mandatory=True, argstr='-a %s',
                                    desc='Structural image, typically T1.  If more than one anatomical image is specified, subsequently specified images are used during the segmentation process.  However, only the first image is used in the registration of priors. Our suggestion would be to specify the T1 as the first image.')
    wholetemplate = File(exists=True, mandatory=True, argstr='-e %s',
                                    desc='Anatomical template created using e.g. LPBA40 data set with buildtemplateparallel.sh in ANTs.')
    brainmask = File(exists=True, mandatory=True, argstr='-m %s',
                                    desc='Brain probability mask created using e.g. LPBA40 data set which have brain masks defined, and warped to anatomical template and averaged resulting in a probability image.')
    output_prefix = traits.Str(usedefault=True, argstr='-o %s', mandatory=True, desc='Output directory + file prefix')
    randomseeding = traits.Float(usedefault=True, argstr='-u %f', desc='Use random number generated from system clock in Atropos (default = 1)')
    

class antsBrainExtractionOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Skull-Stripped Image')


class antsBrainExtraction(ANTSCommand):
    _cmd = 'antsBrainExtraction.sh'
    input_spec = antsBrainExtractionInputSpec
    output_spec = antsBrainExtractionOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.join(os.getcwd(), self.inputs.output_prefix+'BrainExtractionBrain.nii.gz')
        return outputs

#------------------
# Wrap 'ants/ComposeMultiTransform'
class ComposeInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='%d', 
                                usedefault=True,
                                position=0, desc='image dimension (2 or 3)')
    output_file = File(argstr='%s', desc='composed warp transform',
                    name_source=['output_file'], name_template='%s.nii.gz',
                    position=1, hash_files=False)
    reference = File(argstr='-R %s', desc='referencing image of the composition',
                    name_source=['reference'], name_template='%s.nii.gz',
                    position=2, hash_files=False)
    transform1 = File(argstr='%s', desc='first transform for composition',
                    name_source=['trans1'], name_template='%s.nii.gz',
                    position=3, hash_files=False)
    transform2 = File(argstr='%s', desc='second transform for composition',
                    name_source=['trans2'], name_template='%s.nii.gz',
                    position=4, hash_files=False)

class ComposeOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc='composed transform')

class ComposeMultiTransform(ANTSCommand):
    _cmd = 'ComposeMultiTransform'
    input_spec = ComposeInputSpec
    output_spec = ComposeOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_file'] = os.path.join(os.getcwd(), self.inputs.output_file)
        return outputs


class AvgErrorMapOutputSpec(ErrorMapOutputSpec):
    '''
    Added a new average error field 
    '''
    avgerr = traits.Float(desc="Average distance between volume 1 and 2 of all voxels")
    

class AvgErrorMap(ErrorMap):
    '''
    A simple wrap of nipype.algorithms.metrics.ErrorMap
    to also output an average sum distance (a float) besides the errormap volume
    '''

    output_spec = AvgErrorMapOutputSpec

    def _run_interface(self, runtime):
        # Get two numpy data matrices
        nii_ref = nb.load(self.inputs.in_ref)
        ref_data = np.squeeze(nii_ref.get_data())
        tst_data = np.squeeze(nb.load(self.inputs.in_tst).get_data())
        assert(ref_data.ndim == tst_data.ndim)

        # Load mask
        comps = 1
        mapshape = ref_data.shape

        if (ref_data.ndim == 4):
            comps = ref_data.shape[-1]
            mapshape = ref_data.shape[:-1]

        if isdefined(self.inputs.mask):
            msk = nb.load( self.inputs.mask ).get_data()
            if (mapshape != msk.shape):
                raise RuntimeError("Mask should match volume shape, \
                                   mask is %s and volumes are %s" %
                                   (list(msk.shape), list(mapshape)))
        else:
            msk = np.ones(shape=mapshape)

        # Vectorise both volumes and make the pixel differennce
        mskvector = msk.reshape(-1)
        msk_idxs = np.where(mskvector==1)
        refvector = ref_data.reshape(-1,comps)[msk_idxs].astype(np.float32)
        tstvector = tst_data.reshape(-1,comps)[msk_idxs].astype(np.float32)
        diffvector = (refvector-tstvector)

        # scale the diffrernce
        if self.inputs.metric == 'sqeuclidean':
            errvector = diffvector**2
        elif self.inputs.metric == 'euclidean':
            X = np.hstack((refvector, tstvector))
            errvector = np.linalg.norm(X, axis=1)

        if (comps > 1):
            errvector = np.sum(errvector, axis=1)
        else:
            errvector = np.squeeze(errvector)

        errvectorexp = np.zeros_like(mskvector)
        errvectorexp[msk_idxs] = errvector

        # Get averaged sum error
        self._avgerr = np.average(errvectorexp)

        errmap = errvectorexp.reshape(mapshape)

        hdr = nii_ref.get_header().copy()
        hdr.set_data_dtype(np.float32)
        hdr['data_type'] = 16
        hdr.set_data_shape(mapshape)

        if not isdefined(self.inputs.out_map):
            fname,ext = op.splitext(op.basename(self.inputs.in_tst))
            if ext=='.gz':
                fname,ext2 = op.splitext(fname)
                ext = ext2 + ext
            self._out_file = op.abspath(fname + "_errmap" + ext)
        else:
            self._out_file = self.inputs.out_map

        nb.Nifti1Image(errmap.astype(np.float32), nii_ref.get_affine(),
                       hdr).to_filename(self._out_file)

        return runtime 


    def _list_outputs(self):
        outputs = super(AvgErrorMap, self)._list_outputs()
        outputs['avgerr'] = self._avgerr
        return outputs


