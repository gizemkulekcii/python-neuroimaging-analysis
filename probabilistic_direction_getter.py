from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames, small_sphere
from dipy.direction import ProbabilisticDirectionGetter, peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.viz import actor, colormap, has_fury, window
from scipy.ndimage import binary_dilation, binary_erosion
from dipy.direction import DeterministicMaximumDirectionGetter
import numpy as np
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


'''
# Dilate and erode masks
def refine_mask(mask, iterations=1):
    dilated_mask = binary_dilation(mask, iterations=iterations)
    refined_mask = binary_erosion(dilated_mask, iterations=iterations)
    return refined_mask.astype(bool)

ic_left_mask_data = refine_mask(IC_left_mask)
hg_left_mask_data = refine_mask(HG_left_mask)

ic_right_mask_data = refine_mask(IC_right_mask)
hg_right_mask_data = refine_mask(HG_right_mask)
'''
'''
# Threshold and binarize the hg mask data
hg_right_mask_data = np.where(hg_right_mask_data > 0, 1, 0)
hg_left_mask_data = np.where(hg_left_mask_data > 0, 1, 0)

# Convert to boolean
ac_right_mask = hg_right_mask_data.astype(bool)
ac_left_mask = hg_left_mask_data.astype(bool)
'''

response, ratio = auto_response_ssst(gtab, dwi_data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=6)
csd_fit = csd_model.fit(dwi_data, mask= expanded_mask)

# Seeds from mask
#seed_mask_IC_left = IC_left_mask > 0
seed_mask_IC_right = IC_right_mask > 0

seeds_IC_left = utils.seeds_from_mask(IC_left_mask, dwi_affine, density= [40, 40, 40])
#seeds_IC_right = utils.seeds_from_mask(seed_mask_IC_right, dwi_affine, density=[30, 30, 30])

expanded_hg_left = binary_dilation(HG_left_mask, iterations=7) 

#CSA model for stopping criterion
csa_model = CsaOdfModel(gtab, sh_order_max=6)
gfa = csa_model.fit(dwi_data, mask= expanded_mask).gfa
stopping_criterion = ThresholdStoppingCriterion(gfa, 0.15)

fod = csd_fit.odf(small_sphere)
pmf = fod.clip(min=0)
prob_dg = ProbabilisticDirectionGetter.from_pmf(
    pmf, max_angle=30.0, sphere=small_sphere
)

# Expand HG mask to include a larger region around the auditory cortex
#expanded_hg_left = binary_dilation(HG_left_mask, iterations=5)  # iterations: increase to capture a larger region 
#expanded_hg_right = binary_dilation(hg_right_mask_data, iterations=50)



streamline_generator_left = LocalTracking(prob_dg, stopping_criterion, seeds_IC_left, dwi_affine, step_size=0.5)
#streamline_generator_right = LocalTracking(prob_dg, stopping_criterion, seeds_IC_right, dwi_affine, step_size=0.5)

streamlines_through_mgb_left = target(streamline_generator_left, dwi_affine, TH_left_mask)
'''
streamlines_left = Streamlines([
    s for s in streamlines_through_mgb_left
    if len(s) > 1 and
       np.any(IC_left_mask[tuple(np.round(s[0]).astype(int))]) and  # Start: IC
       np.any(HG_left_mask[tuple(np.round(s[-1]).astype(int))])  # End: Auditory Cortex
])
'''
#streamlines_left = Streamlines(streamline_generator_left)
#streamlines_right = Streamlines(streamline_generator_right)
#streamlines_left = Streamlines([s for s in streamline_generator_left 
 #                               if np.any(expanded_hg_left[tuple(np.round(s[-1]).astype(int))])])

#streamlines_right = Streamlines([s for s in streamline_generator_right 
 #                                if np.any(expanded_hg_right[tuple(np.round(s[-1]).astype(int))])])

sft_left = StatefulTractogram(streamlines_through_mgb_left, dwi_img, Space.RASMM)
#sft_right = StatefulTractogram(streamlines_right, dwi_img, Space.RASMM)
save_trk(sft_left, "deneme2/ic20_left_tractogram_probabilistic_dg_pmf.trk")
#save_trk(sft_right, "deneme2/ic30_right_tractogram_probabilistic_dg_pmf.trk")

'''
if has_fury:
    scene = window.Scene()
    scene.add(actor.line(streamlines_left, colors=colormap.line_colors(streamlines_left)))
    window.record(
        scene=scene, out_path="tractogram_probabilistic_dg_pmf.png", size=(800, 800)
    )
    if interactive:
        window.show(scene)
'''

'''
# the Deterministic Maximum Direction Getter
detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(
    csd_fit.shm_coeff, max_angle=30.0, sphere=default_sphere, sh_to_pmf=True
)
streamline_generator = LocalTracking(
    detmax_dg, stopping_criterion, seeds_IC_left, dwi_affine, step_size=0.5
)
streamlines = Streamlines(streamline_generator)

sft = StatefulTractogram(streamlines, dwi_img, Space.RASMM)
save_trk(sft, "_left_tractogram_deterministic_dg.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(streamlines, colors=colormap.line_colors(streamlines)))
    window.record(
        scene=scene, out_path="tractogram_deterministic_dg.png", size=(800, 800)
    )
    if interactive:
        window.show(scene)
'''