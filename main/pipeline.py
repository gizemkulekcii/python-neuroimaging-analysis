# Pipeline
import os
from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from scipy.ndimage import binary_dilation, binary_erosion
from dipy.tracking import utils
from dipy.tracking.utils import length
from dipy.reconst.shm import CsaOdfModel
from dipy.direction import  peaks_from_model
from dipy.data import default_sphere
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.direction import ProbabilisticDirectionGetter
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines, cluster_confidence
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_trk

def refine_mask(mask, iterations=1): 
    '''Applies dilation followed by erosion to refine binary masks.'''
    dilated_mask = binary_dilation(mask, iterations=iterations)
    refined_mask = binary_erosion(dilated_mask, iterations=iterations)
    return refined_mask.astype(bool)

def seedGeneration(mask_data, dwi_affine, density):
    ''' Generates seeding points for tractography within a given mask.'''
    seeds  = utils.seeds_from_mask(mask= mask_data, affine= dwi_affine, density=density)
    return seeds

def stoppingCriterion_csa(gtab, dwi_data, mask_data): 
    ''' Creates a stopping criterion based on Generalized Fractional Anisotropy (GFA) from CsaOdfModel.'''
    # The CsaOdfModel estimates the fiber orientation distribution function (ODF) from the diffusion data, using spherical harmonics of order 6 for detailed modeling.
    csa_model = CsaOdfModel(gtab = gtab, sh_order_max = 6) # CsaOdfModel for direction estimation
    # Extracts the main diffusion directions from the ODFs across the brain, applying a peak threshold and angular separation to ensure valid fiber directions.
    csa_peaks = peaks_from_model(model = csa_model, data = dwi_data, sphere = default_sphere,
                                      relative_peak_threshold=.8,
                                      min_separation_angle=45,
                                      mask=mask_data)
    stopping_criterion = ThresholdStoppingCriterion(metric_map = csa_peaks.gfa, threshold =.20)
    return stopping_criterion

def probabilisticDirectionGetter_csd(gtab, dwi_data, mask_data): 
    ''' Computes a probabilistic direction getter using Constrained Spherical Deconvolution (CSD)'''
    # Constrained Spherical Deconvolution (CSD) is performed to estimate the fiber orientation distribution with higher accuracy by deconvolving the signal. 
    response, ratio = auto_response_ssst(gtab=gtab, data=dwi_data, roi_radii=10, fa_thr=0.7) # Estimates the response function from the data
    csd_model = ConstrainedSphericalDeconvModel(gtab=gtab, response=response, sh_order_max=6) # Fits the CSD model to the DWI data.
    csd_fit = csd_model.fit(data=dwi_data, mask=mask_data)
    # The ProbabilisticDirectionGetter is initialized with spherical harmonic coefficients from the CSD fit. It probabilistically samples fiber directions during tractography, allowing for modeling of crossing fibers.
    prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff, max_angle=45., sphere=default_sphere)
    return prob_dg


def streamlinesGen(directionGetter, stopping_criterion, seeds, dwi_affine): 
    '''Generates streamlines using probabilistic tractography.'''
    streamlines_generator = LocalTracking(direction_getter = directionGetter, stopping_criterion = stopping_criterion, seeds = seeds, affine = dwi_affine, step_size= 0.5) 
    # The generated streamlines are stored in a Streamlines object, and a StatefulTractogram is created to keep track of the streamlines in a specific coordinate space (RASMM - Right-Anterior-Superior in millimeters).
    streamlines = Streamlines(streamlines_generator)
    sft = StatefulTractogram(streamlines, dwi_img, Space.RASMM) # Convert to StatefulTractogram
    return streamlines, sft

def filterStreamlines(streamlines, dwi_affine, mask, dwi_img): 
    '''Filters streamlines that pass through a specific target mask.'''
    filtered_streamlines = utils.target(streamlines=streamlines , affine= dwi_affine,  target_mask= mask )
    filtered_streams =  Streamlines(filtered_streamlines)
    filtered_sft = StatefulTractogram(filtered_streams, dwi_img, Space.RASMM)
    return filtered_streams, filtered_sft

def excludeStreamlines(streamlines, dwi_affine, mask, dwi_img): 
    '''Excludes streamlines that pass through a specific mask.'''
    filtered_streamlines = utils.target(streamlines=streamlines , affine= dwi_affine,  target_mask= mask, include= False )
    filtered_streams =  Streamlines(filtered_streamlines)
    filtered_sft = StatefulTractogram(filtered_streams, dwi_img, Space.RASMM)
    return filtered_streams, filtered_sft

def keepStreamlines(streamlines): #
    '''Retains streamlines with high confidence based on clustering and length filtering.'''
    lengths = list(length(streamlines))
    long_streamlines = Streamlines()
    for i, sl in enumerate(streamlines):
        if lengths[i] > 40:
            long_streamlines.append(sl)
    cci = cluster_confidence(long_streamlines)
    keep_streamlines = Streamlines()
    for i, sl in enumerate(long_streamlines):
        if cci[i] >= 1:
            keep_streamlines.append(sl)
    sft = StatefulTractogram(keep_streamlines, dwi_img, Space.RASMM)
    return keep_streamlines, sft


# Define the root directory containing subject folders
root_dir = 'data'  

# Get a list of all subject directories
subject_dirs = [os.path.join(root_dir, subj) for subj in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subj))]

# Loop through each subject directory
for subject_dir in subject_dirs:
    subject_id = os.path.basename(subject_dir)
    print(f"Processing subject: {subject_id}")
    
    # Load Diffusion Weighted Imaging (DWI) data
    dwi_data, dwi_affine, dwi_img = load_nifti(os.path.join(subject_dir, f"{subject_id}_space-ACPC_desc-preproc_dwi.nii"), return_img=True)
    
    # Load gradient table (b-values and b-vectors)
    bval_fname = os.path.join(subject_dir, f"{subject_id}_space-ACPC_desc-preproc_dwi.bval")
    bvec_fname = os.path.join(subject_dir, f"{subject_id}_space-ACPC_desc-preproc_dwi.bvec")
    bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname) # Read bvals and bvecs and create gradient table
    gtab = gradient_table(bvals=bvals, bvecs=bvecs) # Creates a gradient table for diffusion modeling.
    
    # Load hemisphere brain masks
    lhemisphere_mask, lhemisphere_mask_affine, lhemisphere_mask_img = load_nifti( os.path.join(subject_dir,"left_hemisphere_brain_mask.nii"), return_img=True)
    rhemisphere_mask, rhemisphere_mask_affine, rhemisphere_mask_img = load_nifti(os.path.join(subject_dir,"right_hemisphere_brain_mask.nii"), return_img=True)
    
    lhemisphere_mask = refine_mask(lhemisphere_mask)
    rhemisphere_mask = refine_mask(rhemisphere_mask)

    # Load masks for Internal Capsule (IC), Heschl’s Gyrus (HG), and Frontal regions and refine masks
    ic_left_mask, ic_left_mask_affine = load_nifti(os.path.join(subject_dir, f"{subject_id}_ic_left_acpc.nii"))
    ic_right_mask, ic_right_mask_affine = load_nifti(os.path.join(subject_dir, f"{subject_id}_ic_right_acpc.nii"))
    hg_left_mask, hg_left_mask_affine = load_nifti(os.path.join(subject_dir, f"{subject_id}_hg_left_acpc.nii"))
    hg_right_mask, hg_right_mask_affine = load_nifti(os.path.join(subject_dir, f"{subject_id}_hg_right_acpc.nii"))
    frontal_left_mask, frontal_left_affine =load_nifti(os.path.join(subject_dir, f"{subject_id}_frontal_left.nii"))
    frontal_right_mask, frontal_right_affine =load_nifti(os.path.join(subject_dir, f"{subject_id}_frontal_right.nii"))

    ic_left_mask = refine_mask(ic_left_mask)
    ic_right_mask = refine_mask(ic_right_mask)
    hg_left_mask = refine_mask(hg_left_mask)
    hg_right_mask = refine_mask(hg_right_mask)
    frontal_left_mask = refine_mask(frontal_left_mask)
    frontal_right_mask = refine_mask(frontal_right_mask)

    #expanded_ic_left_mask = binary_dilation(ic_left_mask, iterations=1) 
    #expanded_ic_right_mask = binary_dilation(ic_right_mask, iterations=1) 
    #expanded_ac_left_mask = binary_dilation(hg_left_mask, iterations=1) 
    #expanded_ac_right_mask = binary_dilation(hg_right_mask, iterations=1) 

    # Define exclusion masks (frontal regions)
    exclude_mask_left = frontal_left_mask
    exclude_mask_right = frontal_right_mask
    
    # Generate seed points for tractography
    print("Generating IC and AC Seeds...")
    
    ic_left_seeds = seedGeneration(ic_left_mask, dwi_affine,[30, 30, 30])
    ic_right_seeds = seedGeneration(ic_right_mask, dwi_affine,[30, 30, 30])

    ac_left_seeds = seedGeneration(hg_left_mask, dwi_affine,[20, 20, 20])
    ac_right_seeds = seedGeneration(hg_right_mask, dwi_affine,[20, 20, 20])
    
    # Compute stopping criteria for tractography
    print("Generating Threshold Stopping Criterion for left and right hemisphere...")
    stopping_criterion_left = stoppingCriterion_csa(gtab, dwi_data, lhemisphere_mask)
    stopping_criterion_right = stoppingCriterion_csa(gtab, dwi_data, rhemisphere_mask)
    
    # Generate probabilistic direction getter
    print("Generating Probabilistic direction getter for left and right hemisphere...")
    probabilistic_direction_getter_left = probabilisticDirectionGetter_csd(gtab, dwi_data, lhemisphere_mask)
    probabilistic_direction_getter_right = probabilisticDirectionGetter_csd(gtab, dwi_data, rhemisphere_mask)
    
    # Generate streamlines from IC seeds
    print("Generating IC Left and IC Right Streamlines...")
    ic_left_streams, ic_left_sft = streamlinesGen(probabilistic_direction_getter_left, stopping_criterion_left, ic_left_seeds, dwi_affine)
    ic_right_streams, ic_right_sft = streamlinesGen(probabilistic_direction_getter_right, stopping_criterion_right, ic_right_seeds, dwi_affine)

    #save_trk(ic_left_sft, os.path.join(subject_dir, f"{subject_id}_ic301_left_streams_ss05_sc20_pdg45.trk")) 
    #save_trk(ic_right_sft, os.path.join(subject_dir, f"{subject_id}_ic301_right_streams_ss05_sc20_pdg45.trk"))
    
    # Generate streamlines from AC seeds
    print("Generating AC Left and AC Right Streamlines...")
    ac_left_streams, ac_left_sft = streamlinesGen(probabilistic_direction_getter_left, stopping_criterion_left , ac_left_seeds, dwi_affine)
    ac_right_streams, ac_right_sft = streamlinesGen(probabilistic_direction_getter_right, stopping_criterion_right, ac_right_seeds, dwi_affine)

    #save_trk(ac_left_sft, os.path.join(subject_dir, f"{subject_id}_ac20_left_streams_ss05_sc20_pdg45.trk"))
    #save_trk(ac_right_sft, os.path.join(subject_dir, f"{subject_id}_ac20_right_streams_ss05_sc20_pdg45.trk"))
    
    # Filter IC streamlines that reach AC
    print("Filtering IC to AC Streamlines...")
    ic_to_ac_left_streams, ic_to_ac_left_sft  = filterStreamlines(ic_left_streams, dwi_affine, hg_left_mask, dwi_img)
    ic_to_ac_right_streams, ic_to_ac_right_sft  = filterStreamlines(ic_right_streams, dwi_affine, hg_right_mask, dwi_img)

    final_ic_to_ac_left_streams, final_ic_to_ac_left_sft  = excludeStreamlines(ic_to_ac_left_streams, dwi_affine, exclude_mask_left, dwi_img)
    final_ic_to_ac_right_streams, final_ic_to_ac_right_sft  = excludeStreamlines(ic_to_ac_right_streams, dwi_affine, exclude_mask_right, dwi_img)

    #save_trk(final_ic_to_ac_left_sft,  os.path.join(subject_dir, f"{subject_id}_ic30_to_ac_left_streams_ss05_sc20_pdg45.trk"))
    #save_trk(final_ic_to_ac_right_sft, os.path.join(subject_dir, f"{subject_id}_ic30_to_ac_right_streams_ss05_sc20_pdg45.trk"))
    
    # Filter AC streamlines that reach IC
    print("Filtering AC to IC Streamlines...")
    ac_to_ic_left_streams, ac_to_ic_left_sft  = filterStreamlines(ac_left_streams, dwi_affine, ic_left_mask, dwi_img)
    ac_to_ic_right_streams, ac_to_ic_right_sft  = filterStreamlines(ac_right_streams, dwi_affine, ic_right_mask, dwi_img)

    final_ac_to_ic_left_streams, final_ac_to_ic_left_sft = excludeStreamlines(ac_to_ic_left_streams, dwi_affine, exclude_mask_left, dwi_img)
    final_ac_to_ic_right_streams, final_ac_to_ic_right_sft = excludeStreamlines(ac_to_ic_right_streams, dwi_affine, exclude_mask_right, dwi_img)

    #save_trk(final_ac_to_ic_left_sft, os.path.join(subject_dir, f"{subject_id}_ac20_to_ic_left_streams_ss05_sc20_pdg45.trk"))
    #save_trk(final_ac_to_ic_right_sft, os.path.join(subject_dir, f"{subject_id}_ac20_to_ic_right_streams_ss05_sc20_pdg45.trk"))
    
    # Combine IC ↔ AC streamlines
    streamlines1_left = list(final_ic_to_ac_left_streams)
    streamlines2_left = list(final_ac_to_ic_left_streams)
    streamlines1_right = list(final_ic_to_ac_right_streams)
    streamlines2_right = list(final_ac_to_ic_right_streams)

    combined_streamlines_left = Streamlines(streamlines1_left + streamlines2_left)
    combined_left_sft = StatefulTractogram(combined_streamlines_left, dwi_img, Space.RASMM) # Convert to StatefulTractogram

    combined_streamlines_right = Streamlines(streamlines1_right + streamlines2_right)
    combined_right_sft = StatefulTractogram(combined_streamlines_right, dwi_img, Space.RASMM) # Convert to StatefulTractogram

    #save_trk(combined_sft, os.path.join(subject_dir, f"{subject_id}_ic30_ac20_combined_streams_left_ss05_sc20_pdg45.trk")) # Save to a single .trk file
    #save_trk(combined_right_sft, os.path.join(subject_dir, f"{subject_id}_ic30_ac20_combined_streams_right_ss05_sc20_pdg45.trk"))

    keep_streamlines_left, keep_streamlines_left_sft = keepStreamlines(combined_streamlines_left)
    keep_streamlines_right, keep_streamlines_right_sft = keepStreamlines(combined_streamlines_right)

    # Save final combined streamlines
    save_trk(keep_streamlines_left_sft, os.path.join(subject_dir, f"{subject_id}_ic30_ac20_combined_streams_left_ss05_sc20_pdg45.trk"))
    print(f"Saved combined Left IC ↔ AC streamlines for {subject_id}!")
    save_trk(keep_streamlines_right_sft, os.path.join(subject_dir, f"{subject_id}_ic30_ac20_combined_streams_right_ss05_sc20_pdg45.trk"))
    print(f"Saved combined Right IC ↔ AC streamlines for {subject_id}!")