import os
from nipype.interfaces.fsl.preprocess import FSLCommandInputSpec, FSLCommand
from nipype.interfaces.ants.registration import ANTSCommand, ANTSCommandInputSpec, RegistrationOutputSpec
from nipype.interfaces.base import TraitedSpec, File, traits, CommandLine, InputMultiPath
from nipype.interfaces.traits_extension import isdefined

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