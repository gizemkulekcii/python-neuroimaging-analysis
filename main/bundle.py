import os
import numpy as np
from dipy.io.image import load_nifti
from dipy.io.streamline import load_trk
from dipy.tracking.streamline import Streamlines
from dipy.segment.clustering import QuickBundles
from dipy.io.pickles import save_pickle
from fury import colormap 
from dipy.viz import window, actor, colormap
from dipy.tracking.streamline import set_number_of_points
#from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric

# Function to cluster streamlines into bundles using QuickBundles algorithm
def bundleStreamlines(streamlines):
    # Resample streamlines to ensure they have the same number of points
    streamlines = set_number_of_points(streamlines, nb_points=12)
    
    # Define metric for clustering (average Euclidean distance between points)
    metric = AveragePointwiseEuclideanMetric()
    
    # Apply QuickBundles clustering with a threshold of 10mm
    qb = QuickBundles(threshold=10., metric= metric)
    clusters = qb.cluster(streamlines)
    
    print(f"Number of clusters found: {len(clusters)}")
    return clusters

# Define root directory containing subject data
root_dir = 'data' 

# Get list of subject directories within root directory
subject_dirs = [os.path.join(root_dir, subj) for subj in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subj))]

# Iterate through each subject's data
for subject_dir in subject_dirs:
    subject_id = os.path.basename(subject_dir)
    print(f"Processing subject: {subject_id}")
    
    # Load diffusion MRI data
    dwi_fname = os.path.join(subject_dir, f"{subject_id}_space-ACPC_desc-preproc_dwi.nii") 
    dwi_data, dwi_affine, dwi_img = load_nifti(dwi_fname, return_img=True)
    dwi_shape = dwi_img.shape[:3] # Extract 3D shape of the image
    
    # Load and process left auditory pathway tractography
    trk_left = load_trk(os.path.join(subject_dir, f"{subject_id}_ic30_ac20_combined_streams_left_ss05_sc20_pdg45.trk"), reference= dwi_fname)
    streamlines_left = Streamlines(trk_left.streamlines)
    bundle_left = bundleStreamlines(streamlines_left)
    print(f"Streamlines for left after QB:{bundle_left}")

    # Save clustered streamlines as a pickle file
    save_pickle(os.path.join(subject_dir, f"{subject_id}_QB_left.pkl"), bundle_left)
    
    # Load and process right auditory pathway tractography
    trk_right = load_trk(os.path.join(subject_dir, f"{subject_id}_ic30_ac20_combined_streams_right_ss05_sc20_pdg45.trk"), reference= dwi_fname)
    streamlines_right = Streamlines(trk_right.streamlines)
    bundle_right = bundleStreamlines(streamlines_right) # Cluster streamlines into bundles
    print(f"Streamlines for right after QB:{bundle_right}")

    # Save clustered streamlines as a pickle file
    save_pickle(os.path.join(subject_dir, f"{subject_id}_QB_right.pkl"), bundle_right)
    
    # Enable visualization
    interactive = False  # Set to True if visualization is needed
    scene = window.Scene()
    
    # Visualize left auditory pathway bundles
    colormap_left = colormap.create_colormap(np.arange(len(bundle_left)))
    scene.clear()
    scene.SetBackground(1, 1, 1)
    scene.add(actor.streamtube(streamlines_left, window.colors.white, opacity=0.05)) # Original streamlines
    scene.add(actor.streamtube(bundle_left.centroids, colormap_left, linewidth=0.4)) # Clustered bundles

    # Save visualization as image
    window.record(scene, out_path=os.path.join(subject_dir, f"{subject_id}_left_bundles.png"), size=(600, 600))
    
    # Show interactive visualization if enabled
    if interactive:
        window.show(scene)
   
    # Visualize right auditory pathway bundles
    colormap_right = colormap.create_colormap(np.arange(len(bundle_right)))
    scene.clear()
    scene.SetBackground(1, 1, 1)
    scene.add(actor.streamtube(streamlines_right, window.colors.white, opacity=0.05)) # Original streamlines
    scene.add(actor.streamtube(bundle_right.centroids, colormap_right, linewidth=0.4)) # Clustered bundles

    # Save visualization as image
    window.record(scene, out_path=os.path.join(subject_dir, f"{subject_id}_right_bundles.png"), size=(600, 600))
    
    # Show interactive visualization if enabled
    if interactive:
        window.show(scene)
 


