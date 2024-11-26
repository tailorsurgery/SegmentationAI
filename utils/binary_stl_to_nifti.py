import os
import numpy as np
import nibabel as nib
import pydicom
from stl import mesh
from skimage.transform import resize
import matplotlib.pyplot as plt

def load_dicom_images(dicom_directory, target_shape=(512, 512)):
    """Load DICOM files from a directory, resize to target shape, and return a 3D numpy array."""
    dicom_files = [f for f in os.listdir(dicom_directory) if f.endswith('.dcm')]
    dicom_files.sort()  # Sort files to maintain order
    slices = []

    for file in dicom_files:
        file_path = os.path.join(dicom_directory, file)
        dicom_slice = pydicom.dcmread(file_path)
        
        # Get the pixel array
        pixel_array = dicom_slice.pixel_array

        # Print original shape for debugging
        print(f'Original shape of slice {file}: {pixel_array.shape}')

        # Check if the pixel array is 2D
        if len(pixel_array.shape) == 2:
            # Resize the pixel array to the target shape
            resized_slice = resize(pixel_array, target_shape, anti_aliasing=True)
            slices.append(resized_slice)
        else:
            print(f'Skipping slice {file} due to unexpected shape: {pixel_array.shape}')

    if not slices:
        raise ValueError("No valid DICOM slices were loaded.")

    # Convert list of slices into a 3D numpy array
    volume = np.stack(slices, axis=-1)
    return volume

def load_stl_as_volume(stl_file, target_shape):
    """Load a binary STL file as a volume mask and return a 3D numpy array."""
    # Load STL mesh
    stl_mesh = mesh.Mesh.from_file(stl_file)
    
    # Create an empty volume based on target shape
    volume = np.zeros(target_shape, dtype=np.uint8)
    
    # Iterate through each triangle in the mesh and mark the corresponding volume
    for i in range(len(stl_mesh.vectors)):
        # Here we use a simple method to mark the triangles in the volume
        # You might want to use a more sophisticated approach based on your needs
        points = stl_mesh.vectors[i]
        # Normalize points to fit within the volume dimensions
        points_normalized = ((points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0)) * np.array(target_shape)).astype(int)

        # Assuming the points form a triangular shape in the volume
        for point in points_normalized:
            if all(0 <= coord < dim for coord, dim in zip(point, target_shape)):
                volume[point[0], point[1], point[2]] = 1  # Set the voxel to 1 (or another value for a different mask)

    return volume

def save_nifti(volume, filename):
    """Save a 3D numpy array as a NIfTI file."""
    nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))
    nib.save(nifti_img, filename)

def visualize_slice(volume, slice_index):
    """Visualize a specific slice of the 3D volume."""
    plt.imshow(volume[:, :, slice_index], cmap='gray')
    plt.axis('off')
    plt.show()

def main(case):
    """Main function to load DICOM images, create masks from STL, and save them."""
    
    # Define directory paths based on case
    dicom_directory = f'./../TS_DATASET/dataset/dicom/{case}'  # DICOM directory
    stl_directory = f'./../TS_DATASET/dataset/binary_stl/{case}'  # STL directory
    output_dir = f'./../TS_DATASET/dataset/masks'  # Output directory for masks

    # Load DICOM images
    dicom_volume = load_dicom_images(dicom_directory)

    # Initialize a mask index for naming output files
    mask_index = 0

    # Load STL masks and save them as separate NIfTI files
    for stl_file in os.listdir(stl_directory):
        if stl_file.endswith('.stl'):
            stl_path = os.path.join(stl_directory, stl_file)
            print(f'Processing STL file: {stl_path}')
            
            # Load STL as volume
            stl_mask = load_stl_as_volume(stl_path, dicom_volume.shape)

            # Save the individual mask as a NIfTI file
            mask_output_file = os.path.join(output_dir, f'mask_{case}_{mask_index}.nii')
            save_nifti(stl_mask, mask_output_file)

            # Visualize a middle slice of the STL mask
            middle_slice_index = dicom_volume.shape[2] // 2
            visualize_slice(dicom_volume, middle_slice_index)
            visualize_slice(stl_mask, middle_slice_index)

            # Increment the mask index for the next output file
            mask_index += 1

if __name__ == '__main__':
    case = '240002'  # Change this to your case identifier
    main(case)
