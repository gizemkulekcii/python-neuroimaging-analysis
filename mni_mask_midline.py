import nibabel as nib
import numpy as np
from dipy.io.image import load_nifti
def create_mask(image_path, intensity):

    # Load the NIfTI image using nibabel 
    mask_data, mask_affine, mask_img = load_nifti(image_path, return_img=True)

    # Create the white matter mask 
    mask = (mask_data == intensity)
    
    print("The mask created.  Shape:", mask.shape)

    return mask, mask_affine, mask_img 

def midline(data, affine, img):
    header = img.header
    midline = data.shape[0] // 2

    # Create masks
    left_mask = np.zeros_like(data)
    right_mask = np.zeros_like(data)

    # Assign hemispheres
    left_mask[midline:, :, :] = data[midline:, :, :]
    right_mask[:midline, :, :] = data[:midline, :, :]
    
    # Save the masks as NIfTI files
    left_img = nib.Nifti1Image(left_mask, affine, header)
    right_img = nib.Nifti1Image(right_mask, affine, header)

    print("Left and right hemisphere masks created.")
    return left_mask, right_mask, left_img, right_img


'''
MNI-maxprob-thr25-2mm:
Region: Intensity Value, Caudate:1, Cerebellum: 2, Frontal Lobe: 3, Insula: 4,
Occipital Lobe: 5, Parietal Lobe: 6, Putamen: 7, Temporal Lobe: 8, Thalamus: 9
'''

image_path = "MNI-maxprob-thr25-2mm.nii"  
intensity = 3 

# Create the mask
mask, affine, img = create_mask(image_path, intensity)

output_path="frontal_lobe_mni.nii"
# Save the NIfTI image
nib.save(img, output_path)

left_hemisphere_mask, right_hemisphere_mask, left_hemisphere_img, right_hemisphere_img = midline(mask, affine, img)

output_path_left = "frontal_lobe_left_mni.nii"
output_path_right = "frontal_lobe_right_mni.nii"

nib.save(left_hemisphere_img, output_path_left)
nib.save(right_hemisphere_img, output_path_right)
