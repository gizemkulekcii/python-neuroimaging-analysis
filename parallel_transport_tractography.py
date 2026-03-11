from dipy.direction import peaks_from_model
from dipy.data import default_sphere
from dipy.io.streamline import save_trk
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.data import get_sphere
from dipy.direction import PTTDirectionGetter
from dipy.reconst.shm import CsaOdfModel
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response_ssst)
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.viz import window, actor, colormap, has_fury
from scipy.ndimage import binary_dilation, binary_erosion
import numpy as np
from dipy.tracking.utils import target

dwi_fname = 'deneme2/sub-hno002_space-ACPC_desc-preproc_dwi.nii'
bval_fname = 'deneme2/sub-hno002_space-ACPC_desc-preproc_dwi.bval'
bvec_fname = 'deneme2/sub-hno002_space-ACPC_desc-preproc_dwi.bvec'
wm_mask_fname = 'deneme2/white_matter2.nii'

dwi_data, dwi_affine, dwi_img = load_nifti(dwi_fname, return_img=True)
wm_mask_data, wm_mask_affine, wm_mask_img = load_nifti(wm_mask_fname, return_img=True)

bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)
gtab = gradient_table(bvals, bvecs)

IC_left_mask_fname = "deneme2/ic_left_acpc.nii"
IC_right_mask_fname = "deneme2/ic_right_acpc.nii"
HG_right_mask_fname = "deneme2/hg_right_acpc.nii"
HG_left_mask_fname = "deneme2//hg_left_acpc.nii"
mT_mask_fname ="deneme2/mgb.nii"

IC_left_mask, IC_left_mask_affine = load_nifti(IC_left_mask_fname)
IC_right_mask, IC_right_mask_affine = load_nifti(IC_right_mask_fname)
HG_left_mask, HG_left_mask_affine = load_nifti(HG_left_mask_fname)
HG_right_mask, HG_right_mask_affine = load_nifti(HG_right_mask_fname)
mT_mask, mT_affine = load_nifti(mT_mask_fname)

ic_left_mask_data = binary_dilation(IC_left_mask, iterations= 1).astype(bool)
hg_left_mask_data = binary_dilation(HG_left_mask, iterations= 1).astype(bool)

ic_right_mask_data = binary_dilation(IC_right_mask, iterations= 1).astype(bool)
hg_right_mask_data = binary_dilation(HG_right_mask, iterations= 1).astype(bool)

mT_mask = dilated_mask = binary_dilation(mT_mask, iterations= 1).astype(bool)

response, ratio = auto_response_ssst(gtab, dwi_data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=6)
csd_fit = csd_model.fit(dwi_data, mask=wm_mask_data)

# Seeds from mask
seed_mask_IC_left = ic_left_mask_data > 0
seed_mask_IC_right = ic_right_mask_data > 0

seeds_IC_left = utils.seeds_from_mask(seed_mask_IC_left, dwi_affine, density= [20, 20, 20])
#seeds_IC_right = utils.seeds_from_mask(seed_mask_IC_right, dwi_affine, density=[40, 40, 40])

#CSA model for stopping criterion
csa_model = CsaOdfModel(gtab, sh_order_max=6)
gfa = csa_model.fit(dwi_data, mask=wm_mask_data).gfa
stopping_criterion = ThresholdStoppingCriterion(gfa, 0.25)

# Prepare the PTT direction getter using the fiber ODF (FOD) obtain with CSD. 
# Start the local tractography using PTT direction getter.

sphere = get_sphere(name='repulsion724')
fod = csd_fit.odf(sphere)
pmf = fod.clip(min=0)
ptt_dg = PTTDirectionGetter.from_pmf(pmf, max_angle=15, probe_length=0.5,
                                     sphere=sphere)
'''
# Expand HG mask to include a larger region around the auditory cortex
expanded_hg_left = binary_dilation(hg_left_mask_data, iterations=10)  # iterations: increase to capture a larger region 
expanded_hg_right = binary_dilation(hg_right_mask_data, iterations=10)
'''
# Parallel Transport Tractography
streamline_generator_left = LocalTracking(direction_getter=ptt_dg,
                                     stopping_criterion=stopping_criterion,
                                     seeds=seeds_IC_left,
                                     affine=dwi_affine,
                                     step_size=0.2)
streamlines_left = Streamlines(streamline_generator_left)

'''
streamline_generator_right = LocalTracking(direction_getter=ptt_dg,
                                     stopping_criterion=stopping_criterion,
                                     seeds=seeds_IC_right,
                                     affine=dwi_affine,
                                     step_size=0.2)
streamlines_right = Streamlines(streamline_generator_right)
'''
'''
# Filter streamlines that end in or near the expanded auditory cortex region
streamlines_left = Streamlines([s for s in streamline_generator_left 
                                if np.any(expanded_hg_left[tuple(np.round(s[-1]).astype(int))])])

streamlines_right = Streamlines([s for s in streamline_generator_right 
                                 if np.any(expanded_hg_right[tuple(np.round(s[-1]).astype(int))])])
'''
'''
streamlines_through_mgb_left = target(streamline_generator_left, dwi_affine, mT_mask)
streamlines_through_mgb_right = target(streamline_generator_right, dwi_affine, mT_mask)

streamlines_left = Streamlines([
    s for s in streamlines_through_mgb_left
    if len(s) > 1 and
       np.any(IC_left_mask[tuple(np.round(s[0]).astype(int))]) and  # Start: IC
       np.any(HG_left_mask[tuple(np.round(s[-1]).astype(int))])  # End: Auditory Cortex
])

streamlines_right = Streamlines([
    s for s in streamlines_through_mgb_right
    if len(s) > 1 and
       np.any(IC_right_mask[tuple(np.round(s[0]).astype(int))]) and  # Start: IC
       np.any(HG_right_mask[tuple(np.round(s[-1]).astype(int))])  # End: Auditory Cortex
])
'''
'''
# Stop at MGB and continue to Auditory Cortex
streamlines_left = Streamlines([
    s for s in streamline_generator_left 
    if len(s) > 0 and
       np.any(mT_mask[tuple(np.round(s[-1]).astype(int))]) and  # Passes through MGB
       np.any(HG_left_mask[tuple(np.round(s[-1]).astype(int))])  # Ends at Auditory Cortex
])
streamlines_right = Streamlines([
    s for s in streamline_generator_right
    if len(s) > 0 and
       np.any(mT_mask[tuple(np.round(s[-1]).astype(int))]) and  # Passes through MGB
       np.any(HG_right_mask[tuple(np.round(s[-1]).astype(int))])  # Ends at Auditory Cortex
])
'''
sft = StatefulTractogram(streamlines_left, dwi_img, Space.RASMM)
save_trk(sft, "left20_ic_mT_ac_tractogram_ptt_dg_pmf.trk")
#sft = StatefulTractogram(streamlines_right, dwi_img, Space.RASMM)
#save_trk(sft, "right20_ic_mT_ac_tractogram_ptt_dg_pmf.trk")

'''
if has_fury:
    scene = window.Scene()
    scene.add(actor.line(streamlines, colormap.line_colors(streamlines)))
    window.record(scene, out_path='tractogram_ptt_dg_pmf.png',
                  size=(800, 800))
    if interactive:
        window.show(scene)
'''