#Diffusion Metrics
import os
import nibabel as nib
import array as arr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dipy.io.image import load_nifti
from dipy.viz import window, actor, colormap
from dipy.tracking.streamline import transform_streamlines
import pickle
from scipy.ndimage import map_coordinates

# Function to visualize and save diffusion metric projections (AD, FA, MD, RD)
def visualize_and_save(metric_name, values, bundle, bundle_idx, output_filename):
    try:
        bundle_array = np.array(bundle)  # Convert bundles to numpy array

        # Normalize the diffusion values between 0 and 1 (for colormap mapping)
        values = np.asarray(values) 
        normalized_values = values / np.max(values) 

        # If values are multi-dimensional, flatten them to 1D
        if normalized_values.ndim > 1:
            normalized_values = normalized_values.flatten() 

        # Map the normalized values to a colormap
        colormap_value = colormap.create_colormap(normalized_values)

        scene = window.Scene()
        # Create and add the streamtubes to the scene with the colormap applied
        stream_actor = actor.streamtube(bundle_array, colors=colormap_value)
        scene.add(stream_actor)
        
        # Create a bar for the diffusion values
        bar = actor.scalar_bar( title="Diffusion Values")
        scene.add(bar)
        
        # Manually set the scalar bar range to match the diffusion values
        bar.GetProperty().SetOpacity(1.0) # Ensure the scalar bar is fully visible
        bar.GetLabelTextProperty().SetColor(1, 1, 1)  # Set text color to white
        bar.GetLookupTable().SetRange(np.min(values), np.max(values))  # Set correct range
        
        # Show the visualization
        #window.show(scene)
        window.record(scene, out_path=output_filename, size=(600, 600))
        
        print(f"Saved {metric_name} visualization for bundle {bundle_idx}")
    except Exception as e:
        print(f"Error in visualize_and_save: {e}")

# Define root directory where subject data is stored
root_dir = 'data' 
subject_dirs = [os.path.join(root_dir, subj) for subj in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subj))]

# Loop through each subject directory
for subject_dir in subject_dirs:
    subject_id = os.path.basename(subject_dir)
    print(f"Processing subject: {subject_id}")
    
    # Load diffusion metric maps (AD, FA, MD, RD) for the current subject
    ad_fname = os.path.join(subject_dir, f"AD/{subject_id}_AD.nii") 
    ad_data, ad_affine, ad_img = load_nifti(ad_fname, return_img=True) 

    fa_fname = os.path.join(subject_dir, f"FA/{subject_id}_FA.nii") # Fractional Anisotropy
    fa_data, fa_affine, fa_img = load_nifti(fa_fname, return_img=True) 
    
    md_fname = os.path.join(subject_dir, f"MD/{subject_id}_MD.nii") # Mean Diffusivity
    md_data, md_affine, md_img = load_nifti(md_fname, return_img=True) 

    rd_fname = os.path.join(subject_dir, f"RD/{subject_id}_RD.nii") # Radial Diffusivity
    rd_data, rd_affine, rd_img = load_nifti(rd_fname, return_img=True) 
    
    affine = ad_affine # Use AD affine as the reference affine transformation

    # Load precomputed QuickBundles (QB) clustered streamlines for left and right hemispheres
    with open(os.path.join(subject_dir, f"{subject_id}_QB_left.pkl"), "rb") as f:
       bundles_left = pickle.load(f)

    with open(os.path.join(subject_dir, f"{subject_id}_QB_right.pkl"), "rb") as f:
       bundles_right = pickle.load(f)

    all_data_left = [] # Store values for all left hemisphere bundles
    
    # Process each left hemisphere bundle
    for bundle_idx, bundle in enumerate(bundles_left):
        # Transform streamlines to native space
        bundles_left_native = transform_streamlines(bundle, np.linalg.inv(affine))
        
        # Initialize lists to store diffusion values for each bundle
        ad_values_bundle = []
        fa_values_bundle = []
        md_values_bundle = []
        rd_values_bundle = []

        # Extract diffusion metric values for each streamline
        for streamline_idx, streamline in enumerate(bundles_left_native):
            # Map streamline coordinates to the diffusion data
            ad_values_left = map_coordinates(ad_data, np.array(streamline).T, order=1, mode='nearest')
            fa_values_left = map_coordinates(fa_data, np.array(streamline).T, order=1, mode='nearest')
            md_values_left = map_coordinates(md_data, np.array(streamline).T, order=1, mode='nearest')
            rd_values_left = map_coordinates(rd_data, np.array(streamline).T, order=1, mode='nearest')
            
            # Append the values to respective lists
            ad_values_bundle.append(ad_values_left)
            fa_values_bundle.append(fa_values_left)
            md_values_bundle.append(md_values_left)
            rd_values_bundle.append(rd_values_left)
            
            # Save all data for later
            for point_idx, (ad, fa, md, rd) in enumerate(zip(ad_values_left, fa_values_left, md_values_left, rd_values_left)):
                all_data_left.append([subject_id, bundle_idx, streamline_idx, point_idx, ad, fa, md, rd])
        
        # Visualize and save metric projections for each bundle 
        visualize_and_save("AD", ad_values_bundle, bundles_left_native, bundle_idx, os.path.join(subject_dir, f"{subject_id}_ad_bundle{bundle_idx}_left.png"))
        visualize_and_save("FA", fa_values_bundle, bundles_left_native, bundle_idx, os.path.join(subject_dir, f"{subject_id}_fa_bundle{bundle_idx}_left.png"))
        visualize_and_save("MD", md_values_bundle, bundles_left_native, bundle_idx, os.path.join(subject_dir, f"{subject_id}_md_bundle{bundle_idx}_left.png"))
        visualize_and_save("RD", rd_values_bundle, bundles_left_native, bundle_idx, os.path.join(subject_dir, f"{subject_id}_rd_bundle{bundle_idx}_left.png"))

    
    # Save left hemisphere diffusion metrics as a CSV file
    all_data_left_df = pd.DataFrame(all_data_left, columns=["Subject", "Bundle", "Streamline", "Point", "AD", "FA", "MD", "RD"])
    csv_path_left = os.path.join(subject_dir, f"{subject_id}_diffusion_metrics_all_points_mean_left.csv")
    all_data_left_df.to_csv(csv_path_left, index=False, float_format='%.6f')
    print(f"Saved left diffusion metrics for all points to {csv_path_left}")
    
    all_data_right = [] # Store values for all right hemisphere bundles
    
    # Process each right hemisphere bundle
    for bundle_idx, bundle in enumerate(bundles_right):
        # Transform streamlines to native space
        bundles_right_native = transform_streamlines(bundle, np.linalg.inv(affine))
        
        # Initialize lists to store diffusion values for each bundle
        ad_values_bundle = []
        fa_values_bundle = []
        md_values_bundle = []
        rd_values_bundle = []
        
        # Extract diffusion metric values for each streamline
        for streamline_idx, streamline in enumerate(bundles_right_native):
            # Map streamline coordinates to the diffusion data
            ad_values_right = map_coordinates(ad_data, np.array(streamline).T, order=1, mode='nearest')
            fa_values_right = map_coordinates(fa_data, np.array(streamline).T, order=1, mode='nearest')
            md_values_right = map_coordinates(md_data, np.array(streamline).T, order=1, mode='nearest')
            rd_values_right = map_coordinates(rd_data, np.array(streamline).T, order=1, mode='nearest')
            
            # Append the values to respective lists
            ad_values_bundle.append(ad_values_right)
            fa_values_bundle.append(fa_values_right)
            md_values_bundle.append(md_values_right)
            rd_values_bundle.append(rd_values_right)
            
            # Save all data for later
            for point_idx, (ad, fa, md, rd) in enumerate(zip(ad_values_right, fa_values_right, md_values_right, rd_values_right)):
                all_data_right.append([subject_id, bundle_idx, streamline_idx, point_idx, ad, fa, md, rd])
        
        # Visualize and save metric projections for each bundle
        visualize_and_save("AD", ad_values_bundle, bundles_right_native, bundle_idx, os.path.join(subject_dir, f"{subject_id}_ad_bundle{bundle_idx}_right.png"))
        visualize_and_save("FA", np.array(fa_values_bundle), np.array(bundles_right_native), bundle_idx, os.path.join(subject_dir, f"{subject_id}_fa_bundle{bundle_idx}_right.png"))
        visualize_and_save("MD", md_values_bundle, bundles_right_native, bundle_idx, os.path.join(subject_dir, f"{subject_id}_md_bundle{bundle_idx}_right.png"))
        visualize_and_save("RD", rd_values_bundle, bundles_right_native, bundle_idx, os.path.join(subject_dir, f"{subject_id}_rd_bundle{bundle_idx}_right.png"))
    
    # Save right hemisphere diffusion metrics as a CSV file
    all_data_right_df = pd.DataFrame(all_data_right, columns=["Subject", "Bundle", "Streamline", "Point", "AD", "FA", "MD", "RD"])
    csv_path_right = os.path.join(subject_dir, f"{subject_id}_diffusion_metrics_all_points_mean_right.csv")
    all_data_right_df.to_csv(csv_path_right, index=False, float_format='%.6f')
    print(f"Saved right diffusion metrics for all points to {csv_path_right}")