
import numpy as np
import matplotlib.pyplot as plt
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from scipy.ndimage import binary_dilation, binary_erosion
from dipy.reconst.shm import CsaOdfModel
from dipy.direction import peaks_from_model
from dipy.direction.peaks import peaks_from_model
from dipy.data import default_sphere
from dipy.reconst.csdeconv import auto_response_ssst
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.viz import window, actor, has_fury
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion, BinaryStoppingCriterion
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.viz import actor, window, colormap as cmap
from dipy.tracking.utils import target


dwi_fname = 'deneme2/sub-hno002_space-ACPC_desc-preproc_dwi.nii'
bval_fname = 'deneme2/sub-hno002_space-ACPC_desc-preproc_dwi.bval'
bvec_fname = 'deneme2/sub-hno002_space-ACPC_desc-preproc_dwi.bvec'
mask_fname = 'deneme2/left_wm_mask_in_dwi.nii'

dwi_data, dwi_affine, dwi_img = load_nifti(dwi_fname, return_img=True)
mask_data, mask_affine, mask_img = load_nifti(mask_fname, return_img=True)

expanded_mask = binary_dilation(mask_data, iterations=7) 

bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)
gtab = gradient_table(bvals, bvecs)

IC_left_mask_fname = "deneme2/ic_left_acpc.nii"
IC_right_mask_fname = "deneme2/ic_right_acpc.nii"
HG_right_mask_fname = "deneme2/hg_right_acpc.nii"
HG_left_mask_fname = "deneme2/hg_left_acpc.nii"
TH_right_mask_fname = "deneme2/thalamus_right_acpc.nii"
TH_left_mask_fname = "deneme2/thalamus_left_acpc.nii"

IC_left_mask, IC_left_mask_affine = load_nifti(IC_left_mask_fname)
IC_right_mask, IC_right_mask_affine = load_nifti(IC_right_mask_fname)
HG_left_mask, HG_left_mask_affine = load_nifti(HG_left_mask_fname)
HG_right_mask, HG_right_mask_affine = load_nifti(HG_right_mask_fname)
TH_left_mask, TH_left_mask_affine = load_nifti(TH_left_mask_fname)
TH_right_mask, TH_right_mask_affine = load_nifti(TH_right_mask_fname)


#CSA Model - getting directions from this diffusion data set.
# CSA Model
response, ratio = auto_response_ssst(gtab, dwi_data, roi_radii=10, fa_thr=0.7)
csa_model = CsaOdfModel(gtab, sh_order=6)
csa_peaks = peaks_from_model(csa_model, dwi_data, default_sphere,
                             relative_peak_threshold=0.8,
                             min_separation_angle=45,
                             mask=expanded_mask)

gfa = csa_model.fit(dwi_data, mask=expanded_mask).gfa
stopping_criterion = ThresholdStoppingCriterion(gfa, 0.2)

exclusion_mask = np.copy(expanded_mask)
exclusion_mask[0:30, :, :] = 0

# restricting the fiber tracking to areas with good directionality information
# restrict fiber tracking to those areas where the ODF shows significant restricted diffusion by thresholding on the generalized fractional anisotropy (GFA).
# Inclusion mask as Binary Stopping Criterion
inclusion_criterion = BinaryStoppingCriterion(HG_left_mask)

# Stopping Criterion using GFA of CSA Model
#gfa = csa_model.fit(dwi_data, mask=expanded_mask).gfa
#stopping_criterion_left = ThresholdStoppingCriterion(gfa, 0.2)
#stopping_criterion_right = ThresholdStoppingCriterion(gfa, 0.25)


# Specify where to “seed” (begin) the fiber tracking

# Seeds from mask
seed_mask_IC_left = IC_left_mask> 0
#seed_mask_IC_right = ic_right_mask_data > 0

seeds_IC_left = utils.seeds_from_mask(seed_mask_IC_left, dwi_affine, density= [30, 30, 30])
#seeds_IC_right = utils.seeds_from_mask(seed_mask_IC_right, dwi_affine, density=[10, 10, 10])

# LocalTracking, using the EuDX algorithm
# Creates deterministic set of streamlines using the EuDX algorithm. Deterministic because if you repeat the fiber
# tracking(keeping all the inputs same), you will get exactly the same set of streamlines

# Initialization of LocalTracking. The computation happens in the next step.
# LocalTracking with inclusion and exclusion
streamlines_generator_left = LocalTracking(csa_peaks, stopping_criterion, seeds_IC_left,
                                           affine=dwi_affine, step_size=0.5,
                                           include_mask=HG_left_mask, 
                                           exclude_mask=exclusion_mask)

#streamlines_generator_right = LocalTracking(csa_peaks, stopping_criterion_right, seeds_IC_right,
 #                                     affine=dwi_affine, step_size=.5)

# Generate streamlines object
streamlines_left = Streamlines(streamlines_generator_left)
#streamlines_right = Streamlines(streamlines_generator_right)
# Target the auditory cortex
targeted_streamlines_left = target(streamlines_left, dwi_affine, HG_left_mask)

# Save tractogram
sft_left = StatefulTractogram(targeted_streamlines_left, dwi_img, Space.RASMM)
save_trk(sft_left, "ic_to_auditory_cortex_left.trk", targeted_streamlines_left)


#sft_right = StatefulTractogram(streamlines_right, dwi_img, Space.RASMM)
#save_trk(sft_right, "right_tractogram_EuDX.trk", streamlines_right) 
''''
# Create a streamline actor from the streamlines
streamlines_actor_left = actor.line(streamlines_left, cmap.line_colors(streamlines_left))
streamlines_actor_right = actor.line(streamlines_right, cmap.line_colors(streamlines_right))

# Create a surface actor 
surface_opacity = 0.5
surface_color = [0, 1, 1]

seedroi_actor_left = actor.contour_from_roi(seed_mask_IC_left, dwi_affine,
                                       surface_color, surface_opacity)
seedroi_actor_right = actor.contour_from_roi(seed_mask_IC_right, dwi_affine,
                                       surface_color, surface_opacity)

# Initialize a "Scene" object and add both actors to the rendering.
scene = window.Scene()
scene.add(streamlines_actor_left)
scene.add(seedroi_actor_left)

scene.add(streamlines_actor_right)
scene.add(seedroi_actor_right)

interactive = True
if interactive:
    window.show(scene)

window.record(scene, out_path='contour_from_roi_tutorial.png', size=(1200, 900))
'''