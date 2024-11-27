import os
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt

def load_nifti_image(nifti_file_path, target_shape=(512, 512)):
    """Load NIfTI image from the specified path and resize each slice to the target shape."""
    nifti_image = nib.load(nifti_file_path)
    
    # Get the data from the NIfTI file as a numpy array
    original_volume = nifti_image.get_fdata()

    # Initialize list to store resized slices
    resized_slices = []

    # Resize each slice to the target shape
    for slice_index in range(original_volume.shape[-1]):
        slice_2d = original_volume[:, :, slice_index]

        # Resize the original slice to the target shape
        resized_slice = cv2.resize(slice_2d, target_shape[::-1], interpolation=cv2.INTER_LINEAR)
        resized_slices.append(resized_slice)

    # Stack resized slices to create a 3D volume
    resized_volume = np.stack(resized_slices, axis=-1)

    return original_volume, resized_volume

def get_nifti_dimensions(nifti_file_path):
    """Extract dimensions of the NIfTI file."""
    nifti_image = nib.load(nifti_file_path)
    nifti_data = nifti_image.get_fdata()

    num_slices = nifti_data.shape[-1]
    rows, columns = nifti_data.shape[0], nifti_data.shape[1]

    print(f'Total slices: {num_slices}, Original Rows: {rows}, Original Columns: {columns}')
    return rows, columns, num_slices

def normalize_image(image):
    # Normalize pixel values between 0 and 255
    image_min = np.min(image)
    image_max = np.max(image)
    if image_max > image_min:
        image_normalized = (image - image_min) / (image_max - image_min) * 255
    else:
        image_normalized = np.zeros_like(image)
    return image_normalized.astype(np.uint8)





def view_slices(original_volume, resized_volume, slice_index):
    """Display the original and resized slices side by side."""
    # Apply normalization before displaying
    original_slice = normalize_image(original_volume[:, :, slice_index])
    resized_slice = normalize_image(resized_volume[:, :, slice_index])

    plt.figure(figsize=(12, 6))

    # Original slice
    plt.subplot(1, 2, 1)
    plt.imshow(original_slice, cmap='gray')
    plt.title(f'Original Slice {slice_index}')
    plt.axis('off')

    # Resized slice
    plt.subplot(1, 2, 2)
    plt.imshow(resized_slice, cmap='gray')
    plt.title(f'Resized Slice {slice_index}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
case = '240002'
nifti_file_path = f'./temp_nifti/{case}_images.nii'  # NIfTI file path
target_shape = (512, 512)  # Target dimensions for resizing
from nilearn import plotting
import nibabel as nib

# Load your NIfTI file
nifti_img = nib.load(nifti_file_path)

# Display the image with an interactive web-based viewer
viewer = plotting.view_img(nifti_img)
viewer.open_in_browser()  # This opens the viewer in your default web browser

# Get original dimensions
original_rows, original_columns, num_slices = get_nifti_dimensions(nifti_file_path)

# Load NIfTI image and resize it
original_volume, resized_volume = load_nifti_image(nifti_file_path, target_shape)

# View a specific slice
slice_index = 50  # Change this index to view different slices
view_slices(original_volume, resized_volume, slice_index)


