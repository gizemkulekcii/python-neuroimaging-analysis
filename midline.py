import numpy as np
import matplotlib.pyplot as plt
from dipy.io.image import load_nifti
import nibabel as nib
import os

root_dir = 'data2'  # Change this to your actual root directory
subject_dirs = [os.path.join(root_dir, subj) for subj in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subj))]
for subject_dir in subject_dirs:
    subject_id = os.path.basename(subject_dir)
    print(f"Processing subject: {subject_id}")

    mask_fname = os.path.join(subject_dir, f"{subject_id}_space-ACPC_desc-brain_mask.nii")
    data, affine, mask_img = load_nifti(mask_fname, return_img=True)

    header = mask_img.header

    midline = data.shape[0] // 2

    # Create masks
    left_mask = np.zeros_like(data)
    right_mask = np.zeros_like(data)

    # Assign hemispheres
    right_mask[:midline, :, :] = data[:midline, :, :]
    left_mask[midline:, :, :] = data[midline:, :, :]

    # Save the masks as NIfTI files
    left_img = nib.Nifti1Image(left_mask, affine, header)
    right_img = nib.Nifti1Image(right_mask, affine, header)

    nib.save(left_img,  os.path.join(subject_dir, "left_hemisphere_brain_mask.nii"))
    nib.save(right_img, os.path.join(subject_dir,"right_hemisphere_brain_mask.nii"))

    print("Left and right hemisphere masks saved.")