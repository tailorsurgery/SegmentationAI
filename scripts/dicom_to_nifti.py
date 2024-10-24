import os
import numpy as np
import nibabel as nib
import pydicom
import time
from numpy import pad
from check import check_nii_file
from time import sleep

def is_dicom_file(file_path):
    try:
        pydicom.dcmread(file_path)
        return True
    except Exception:
        return False
def reslice_to_min_dimension(image):
    """
    Reslice the image to the minimum dimension between width and height (for each slice).
    """
    original_shape = image.shape

    # If the image is 2D (one slice), add a new axis to make it 3D (1, height, width)
    if len(original_shape) == 2:  
        image = image[np.newaxis, ...]  
        print(f"Original image shape: {image.shape}")
    
    num_slices = image.shape[0]  # Number of slices
    height, width = original_shape[-2:]  # Get height and width from the last two dimensions
    min_dim = min(height, width)  # Get the minimum dimension for reslicing

    # Initialize output data with the correct shape (num_slices, min_dim, min_dim)
    output_data = np.zeros((num_slices, min_dim, min_dim), dtype=image.dtype)

    for i in range(num_slices):
        slice_image = image[i]  # Get the slice for reslicing
        current_shape = slice_image.shape

        # Ensure the slice image has the correct shape
        if len(current_shape) != 2:
            print(f"Warning: Unexpected shape {current_shape} for slice {i}. Skipping reslice.")
            continue

        # Crop along height if greater than min_dim
        if current_shape[0] > min_dim:
            slice_image = slice_image[(current_shape[0] - min_dim) // 2:(current_shape[0] + min_dim) // 2, :]
        # Crop along width if greater than min_dim
        if current_shape[1] > min_dim:
            slice_image = slice_image[:, (current_shape[1] - min_dim) // 2:(current_shape[1] + min_dim) // 2]

        # Assign the resliced image to the output array
        output_data[i] = slice_image

    print(f"Resliced image shape: {output_data.shape}")
    return output_data


def load_dicom_folder(dicom_files):
    # Create a list to hold tuples of (dicom_file, instance_number)
    dicom_with_instance_numbers = []

    # Read DICOM files and collect their instance numbers
    for dicom_file in dicom_files:
        try:
            dicom = pydicom.dcmread(dicom_file)
            instance_number = getattr(dicom, 'InstanceNumber', None)
            if instance_number is not None:
                dicom_with_instance_numbers.append((dicom_file, int(instance_number)))
            else:
                print(f"Warning: No InstanceNumber for {dicom_file}. Skipping this file.")
        except Exception as e:
            print(f"Error reading DICOM file {dicom_file}: {e}")

    # Sort the DICOM files based on instance number
    dicom_with_instance_numbers.sort(key=lambda x: x[1])
    
    # Initialize variables for processing the sorted files
    num_slices = len(dicom_with_instance_numbers)
    output_data = None
    orientations = []

    for i, (dicom_file, _) in enumerate(dicom_with_instance_numbers):
        try:
            dicom = pydicom.dcmread(dicom_file)
            if 'PixelData' in dicom:
                pixel_spacing = getattr(dicom, 'PixelSpacing', None)
                row_spacing, col_spacing = pixel_spacing if pixel_spacing and len(pixel_spacing) == 2 else (1.0, 1.0)
                print(f"Pixel spacing: {row_spacing}, {col_spacing}")

                orientation = getattr(dicom, 'ImageOrientationPatient', None)
                if orientation and len(orientation) == 6:
                    orientations.append(orientation)
                else:
                    print(f"Warning: ImageOrientationPatient missing for {dicom_file}. Using default orientation.")
                    orientations.append([1, 0, 0, 0, 1, 0])  # Default axial orientation

                grayscale_image = dicom.pixel_array
                print(f"Grayscale image shape: {grayscale_image.shape}")
                image_data = reslice_to_min_dimension(grayscale_image)

                # Initialize output_data after determining the correct shape from the first slice
                if output_data is None:
                    output_data = np.zeros((num_slices, image_data.shape[1], image_data.shape[2]), dtype=np.float32)
                    print(f"Initialized output data with shape: {output_data.shape}")

                # Assign the resliced image to the output array
                if i < len(output_data):  # Ensure index is valid
                    output_data[i] = image_data
                else:
                    print(f"Warning: Index {i} out of bounds for output_data with shape {output_data.shape}.")
                    
            else:
                print(f"Warning: No PixelData found in {dicom_file}.")
        except Exception as e:
            print(f"Error processing {dicom_file}: {e}")

    # Correct orientation if needed
    if output_data is not None and len(output_data) > 1:
        first_orientation = orientations[0]
        if first_orientation[0] != first_orientation[3]:  # Check if orientation mismatch occurs
            output_data = output_data.transpose(0, 2, 1)  # Swap axes if needed
    else:
        print("Warning: No valid output data or insufficient slices.")

    return output_data


def dicom_to_nifti(input_folder, output_file):
    series_data = {}
    
    # Collect DICOM files by SeriesDescription
    for root, dirs, files in os.walk(input_folder):
        dicom_files = [f for f in files if is_dicom_file(os.path.join(root, f))]
        if dicom_files:
            print(f'Processing folder: {root}')
            
            for dicom_file in dicom_files:
                dicom_path = os.path.join(root, dicom_file)
                try:
                    dicom = pydicom.dcmread(dicom_path)
                    series_description = getattr(dicom, 'SeriesDescription', 'Unknown Series')  # Default to 'Unknown Series'
                    
                    if series_description not in series_data:
                        series_data[series_description] = []
                    series_data[series_description].append(dicom_path)
                except Exception as e:
                    print(f"Error reading DICOM file {dicom_file}: {e}")

    # Process each series separately and save as NIfTI
    for series_description, dicom_files in series_data.items():
        print(f"Processing series: {series_description}")
        
        data = load_dicom_folder(dicom_files)
        
        if data is not None and len(data) > 1:  # Ignore series with only one slice
            nifti_img = nib.Nifti1Image(data, np.eye(4))
            output_file_series = output_file.replace('.nii', f'_{series_description}.nii')  # Save by series description
            nib.save(nifti_img, output_file_series)
            print(f"Saved NIfTI file for series '{series_description}' to: {output_file_series}")
        else:
            print(f"Skipping series '{series_description}' due to insufficient slices.")
    
    if not series_data:
        raise ValueError(f'No valid DICOM data found in {input_folder}.')

# Main processing function
def main_function(case, option):
    print(f"**** Processing case {case}...")
    try:
        input_path = f'./{case}'
        output_path = f'./../TS_DATASET/dataset/images/{case}_images.nii'

        time_start = time.time()
        dicom_to_nifti(input_path, output_path)
        time_end = time.time()
        print(f"Total time taken: {time_end - time_start:.2f} seconds")

        check_nii_file(case, option)
    except Exception as e:
        print(f"Error processing case {case}: {e}")

# Call the main function
for case in ['240097']:   
    main_function(case, 'image')
