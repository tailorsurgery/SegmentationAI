import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pydicom



def check_nii_file(case, option):
    print(f"****Checking case {case}...")
    if option == 'mask':

        # Load the NIfTI file
        nifti_file_path = f'C:\\Users\\samya\\Downloads\\RESEGMENTED\\dataset\\masks\\{case}.nii'  # Update this path
        nifti_img = nib.load(nifti_file_path)
        data = nifti_img.get_fdata()

        # Choose a mask index (e.g., 0 for the first mask)
        mask_index = 0
        mask_data = data[mask_index]

        # Display a middle slice of the 3D mask
        slice_index = mask_data.shape[0] // 2  # Middle slice along the z-axis
        plt.imshow(mask_data[slice_index+5, :, :], cmap='gray')
        plt.title(f'Mask {mask_index + 1} - Slice {slice_index}')
        plt.axis('off')
        plt.show()

    elif option == 'image':

        nifti_file = f'C:\\Users\\samya\\Downloads\\RESEGMENTED\\dataset\\images\\{case}.nii'
        nifti_img = nib.load(nifti_file)
        # Get the image data
        img_data = nifti_img.get_fdata(dtype=np.float32)

        print("Data Shape:", nifti_img.shape)
        
        # Visualize the middle slice
        middle_slice_index = img_data.shape[0] // 2
        plt.imshow(img_data[middle_slice_index, :, :], cmap='gray')
        plt.title(f'Middle Slice (Index {middle_slice_index})')
        plt.axis('off')
        plt.show()

        print("Unique values in the masks data:", np.unique(nifti_img.get_fdata()))
check_nii_file('240002_images_anonymized', 'images')