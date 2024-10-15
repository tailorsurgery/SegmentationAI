import os
import numpy as np
import nibabel as nib
import pydicom
import time
from skimage.transform import resize
from check import check_nii_file

case = 240044 # TODO
option = 'image' # 'image' or 'mask'

def is_dicom_file(file_path):
    """Check if a file is a DICOM file by attempting to read it."""
    try:
        pydicom.dcmread(file_path)
        return True
    except Exception:
        return False

def load_dicom_folder(folder_path, target_shape=(512, 512)):
    """Load DICOM files from a folder and return the image data as a 3D NumPy array."""
    dicom_files = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if is_dicom_file(file_path):  # Check if the file is a DICOM file
            dicom_files.append(file_path)  # Store the file path instead of the object

    if not dicom_files:
        print(f'Warning: No DICOM files found in {folder_path}.')
        return None  # Return None if no DICOM files are found

    # Sort files by InstanceNumber if available, otherwise by file order
    try:
        dicom_files.sort(key=lambda x: int(pydicom.dcmread(x).InstanceNumber))
    except AttributeError:
        print("Warning: Some files lack 'InstanceNumber'. Using default order.")

    # Initialize data array
    num_slices = len(dicom_files)
    shape = (num_slices, target_shape[0], target_shape[1])
    data = np.zeros(shape, dtype=np.float32)

    for i, dicom_file in enumerate(dicom_files):
        try:
            dicom = pydicom.dcmread(dicom_file)
            # Check if pixel data is present
            if 'PixelData' in dicom:
                if hasattr(dicom, 'SamplesPerPixel') and dicom.SamplesPerPixel > 1:
                    # Convert RGB to grayscale if necessary
                    rgb_image = dicom.pixel_array
                    grayscale_image = np.dot(rgb_image[...,:3], [0.2989, 0.5870, 0.1140])  # Weighted sum to get grayscale
                    resized_image = resize(grayscale_image, target_shape, mode='reflect', anti_aliasing=True)
                    data[i, :, :] = resized_image
                else:
                    # Resize single-channel images
                    resized_image = resize(dicom.pixel_array, target_shape, mode='reflect', anti_aliasing=True)
                    data[i, :, :] = resized_image
                print(f'Processed {dicom_file}')
            else:
                print(f'Warning: No pixel data in {dicom_file}.')
        except Exception as e:
            print(f'Error processing {dicom_file}: {e}')

    return data

def dicom_to_nifti(input_folder, output_file):
    """Convert DICOM files from multiple folders into a single NIfTI file."""
    all_data = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(input_folder):
        dicom_files = [f for f in files if is_dicom_file(os.path.join(root, f))]  # Check if the files are DICOM
        if dicom_files:
            print(f'Processing folder: {root}')
            data = load_dicom_folder(root)
            if data is not None:  # Only add data if it's not None
                all_data.append(data)
        else:
            print(f'No DICOM files found in {root}.')

    # Check if any data has been loaded
    if not all_data:
        raise ValueError(f'No DICOM data found in any of the subdirectories of {input_folder}.')

    # Combine all loaded data into a single 3D array
    combined_data = np.concatenate(all_data, axis=0)

    # Create a NIfTI image and save it
    nifti_img = nib.Nifti1Image(combined_data, np.eye(4))
    nib.save(nifti_img, output_file)
    print(f'Successfully saved NIfTI file to: {output_file}')

# Main processing section
print(f"****Processing case {case}...")
try:
    # Directory paths
    input_path = f'C:\\Users\\samya\\Downloads\\RESEGMENTED\\01_DCM_{case}'
    output_path = f'C:\\Users\\samya\\Downloads\\RESEGMENTED\\dataset\\images\\{case}.nii'

    # Run the conversion
    time_start = time.time()
    dicom_to_nifti(input_path, output_path)
    time_end = time.time()
    print(f"Total time taken: {time_end - time_start:.2f} seconds")

    # Check the NIfTI file
    check_nii_file(case, option)
except Exception as e:
    print(f"Error processing case {case}: {e}")
