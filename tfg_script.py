import os
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt
### FIRST RESIZE IMAGES SO THEY HAVE THE SAME SHAPE (512, 512) SOME OF THEM HAD (512x858)###
def load_dicom_images(dicom_directory, target_shape=(512, 512)):
    """Load DICOM images from the specified directory and resize them to the target shape."""
    dicom_files = [f for f in os.listdir(dicom_directory) if f.endswith('.dcm')]
    dicom_files.sort()  # Sort files to maintain order

    original_slices = []
    resized_slices = []

    for file in dicom_files:
        file_path = os.path.join(dicom_directory, file)
        dicom_slice = pydicom.dcmread(file_path)
        
        # Read pixel array
        pixel_array = dicom_slice.pixel_array
        
        # Check if the slice is 3D (e.g., RGB) and convert to grayscale if necessary
        if pixel_array.ndim == 3:  # If it has more than 2 dimensions
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2GRAY)

        # Resize the original slice to the target shape
        resized_slice = cv2.resize(pixel_array, target_shape[::-1], interpolation=cv2.INTER_LINEAR)
        
        # Store the original slice and resized slice
        original_slices.append(pixel_array)
        resized_slices.append(resized_slice)

    # Stack the slices into 3D volumes
    # Resize all original slices to the target shape before stacking
    original_volume = np.stack([cv2.resize(slice_, target_shape[::-1], interpolation=cv2.INTER_LINEAR) for slice_ in original_slices], axis=-1)
    resized_volume = np.stack(resized_slices, axis=-1)
    
    return original_volume, resized_volume

def get_dicom_dimensions(dicom_directory):
    """Extract number of rows, columns, and slices from DICOM files in the specified directory."""
    dicom_files = [f for f in os.listdir(dicom_directory) if f.endswith('.dcm')]
    dicom_files.sort()  # Sort files to maintain order

    num_slices = len(dicom_files)
    rows, columns = 0, 0

    for file in dicom_files:
        file_path = os.path.join(dicom_directory, file)
        dicom_slice = pydicom.dcmread(file_path)

        # Extract dimensions
        if 'Rows' in dicom_slice and 'Columns' in dicom_slice:
            rows = dicom_slice.Rows
            columns = dicom_slice.Columns

    print(f'Total slices: {num_slices}, Original Rows: {rows}, Original Columns: {columns}')
    return rows, columns, num_slices

def view_slices(original_volume, resized_volume, slice_index):
    """Display the original and resized slices side by side."""
    original_slice = original_volume[:, :, slice_index]
    resized_slice = resized_volume[:, :, slice_index]

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
dicom_directory = f'./../TS_DATASET/dataset/dicom/{case}'  # DICOM directory
target_shape = (512, 512)  # Target dimensions for resizing

# Get original dimensions
original_rows, original_columns, num_slices = get_dicom_dimensions(dicom_directory)

# Load DICOM images and resize them
original_volume, resized_volume = load_dicom_images(dicom_directory, target_shape)

# View a specific slice
#slice_index = 0  # Change this index to view different slices
#view_slices(original_volume, resized_volume, slice_index)
