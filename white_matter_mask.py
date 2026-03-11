import nibabel as nib
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import affine_transform
from dipy.io.image import load_nifti

#Creates a white matter mask from a NIfTI image based on intensity value.
def create_white_matter_mask(image_path, white_matter_intensity):

    # Load the NIfTI image using nibabel
    img = nib.load(image_path)
    data = img.get_fdata()

    # Create the white matter mask 
    white_matter_mask = (data == white_matter_intensity)

    return white_matter_mask, img.affine # Return both the mask and the affine transformation

if __name__ == "__main__":

    
    anat_mask_path = "deneme2/sub-hno002_space-ACPC_desc-aseg_dseg.nii"  
    anat_mask_data,anat_affine = load_nifti(anat_mask_path)

    left_white_matter_intensity = 2
    right_white_matter_intensity = 41

    # Create the white matter masks
    left_white_matter_mask, affine = create_white_matter_mask(anat_mask_path, left_white_matter_intensity)
    right_white_matter_mask, affine = create_white_matter_mask(anat_mask_path, right_white_matter_intensity)

    print("White matter masks created.")

    mask_img = nib.Nifti1Image(left_white_matter_mask.astype(np.int16), affine)
    output_path="deneme2/left_white_matter_mask_sub2.nii"

    mask_img2 = nib.Nifti1Image(right_white_matter_mask.astype(np.int16), affine)
    output_path2 ="deneme2/right_white_matter_mask_sub2.nii"

    # Save the NIfTI image
    nib.save(mask_img, output_path)
    nib.save(mask_img2, output_path2)
