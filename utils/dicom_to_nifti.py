import os
import numpy as np
import nibabel as nib
import pydicom
import time
from utils.check import check_nii_file

def is_dicom_file(file_path):
    try:
        pydicom.dcmread(file_path)
        return True
    except Exception:
        return False

def reslice_to_aspect_ratio(image):
    """
    Reslice the image while maintaining the aspect ratio.
    Each slice is rescaled based on the aspect ratio calculated from its original dimensions.
    """
    original_shape = image.shape

    # If the image is 2D (one slice), add a new axis to make it 3D (1, height, width)
    if len(original_shape) == 2:
        image = image[np.newaxis, ...]  # Convert to 3D
        #print(f"Original image shape: {image.shape}")

    num_slices = image.shape[0]  # Number of slices
    height, width = original_shape[-2:]  # Get height and width from the last two dimensions

    # Initialize output data with the correct shape
    output_data = np.zeros((num_slices, height, width), dtype=image.dtype)

    for i in range(num_slices):
        slice_image = image[i]  # Get the slice for reslicing
        current_shape = slice_image.shape
        
        if len(current_shape) != 2:
            print(f"Warning: Unexpected shape {current_shape} for slice {i}. Skipping reslice.")
            continue

        # Calculate the aspect ratio
        aspect_ratio = width / height

        # Determine the new dimensions maintaining the aspect ratio
        if aspect_ratio > 1:  # Wider than tall
            new_width = width
            new_height = int(height * (width / height))  # Maintain aspect ratio
        else:  # Taller than wide
            new_height = height
            new_width = int(width * (height / width))  # Maintain aspect ratio

        # Resize the image while maintaining aspect ratio
        if new_height < height:
            slice_image = slice_image[(height - new_height) // 2:(height + new_height) // 2, :]
        if new_width < width:
            slice_image = slice_image[:, (width - new_width) // 2:(width + new_width) // 2]

        # Assign the resliced image to the output array
        output_data[i] = slice_image

    #print(f"Resliced image shape: {output_data.shape}")

    return output_data


def determine_orientation(dicom):
    """ Determine the orientation based on the DICOM ImageOrientationPatient. """
    orientation = dicom.ImageOrientationPatient
    if orientation:
        # Convert orientation to float
        orientation = np.array(orientation, dtype=np.float32)

        if len(orientation) == 6:
            # Extract the two direction vectors
            x_vector = orientation[:3]
            y_vector = orientation[3:]

            # Normalize the vectors
            x_vector_norm = x_vector / np.linalg.norm(x_vector)
            y_vector_norm = y_vector / np.linalg.norm(y_vector)

            # Compute dot products to check orientation
            axial = np.array([0, 0, 1])  # Axial (z-axis)
            coronal = np.array([0, 1, 0])  # Coronal (y-axis)
            sagittal = np.array([1, 0, 0])  # Sagittal (x-axis)

            # Calculate the angle differences
            axial_angle = np.arccos(np.clip(np.dot(x_vector_norm, axial), -1.0, 1.0))
            coronal_angle = np.arccos(np.clip(np.dot(y_vector_norm, coronal), -1.0, 1.0))
            sagittal_angle = np.arccos(np.clip(np.dot(x_vector_norm, sagittal), -1.0, 1.0))

            # Determine which orientation is closest
            if axial_angle < np.pi / 4:  # Less than 45 degrees
                return 'Axial'
            elif coronal_angle < np.pi / 4:
                return 'Coronal'
            elif sagittal_angle < np.pi / 4:
                return 'Sagittal'
            else:
                print(f"Orientation vectors: {x_vector}, {y_vector}")  # Debug print
                return 'Unknown'
        else:
            print(f"Unexpected orientation length: {len(orientation)}")  # Debug print
    return 'Unknown'


def load_dicom_folder(dicom_files):
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
    
    num_slices = len(dicom_with_instance_numbers)
    output_data = None

    for i, (dicom_file, _) in enumerate(dicom_with_instance_numbers):
        try:
            dicom = pydicom.dcmread(dicom_file)
            if 'PixelData' in dicom:
                pixel_spacing = getattr(dicom, 'PixelSpacing', None)
                row_spacing, col_spacing = pixel_spacing if pixel_spacing and len(pixel_spacing) == 2 else (1.0, 1.0)
                #print(f"Pixel spacing: {row_spacing}, {col_spacing}")

                grayscale_image = dicom.pixel_array
                #print(f"Grayscale image shape: {grayscale_image.shape}")

                # Reslice the image to maintain aspect ratio
                image_data = reslice_to_aspect_ratio(grayscale_image)

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
                    series_description = getattr(dicom, 'SeriesDescription', 'Unknown Series')
                    
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
            # Create a dummy affine for initialization (this will be updated later)
            dummy_affine = np.eye(4)

            # Check the DICOM files for orientation and apply appropriate rotation if needed
            for dicom_file in dicom_files:
                dicom = pydicom.dcmread(dicom_file)
                orientation = determine_orientation(dicom)

                # Print detected orientation
                print(f"Detected orientation: {orientation}")

                # Set affine based on orientation
                if orientation == 'Axial':
                    new_affine = dummy_affine  # No rotation needed
                elif orientation == 'Coronal':
                    print("Detected coronal orientation. Adjusting affine.")
                    new_affine = update_affine_with_rotation(dummy_affine, -90)  # Adjust affine for coronal
                elif orientation == 'Sagittal':
                    print("Detected sagittal orientation. Adjusting affine.")
                    new_affine = update_affine_with_rotation(dummy_affine, 90)  # Adjust affine for sagittal
                else:
                    print("Unknown orientation. Keeping default affine.")
                    new_affine = dummy_affine  # Keep the default for unknown orientations

                dummy_affine = new_affine  # Update dummy_affine for further iterations

            # Create NIfTI image and save it
            nifti_img = nib.Nifti1Image(data, dummy_affine)
            output_file_series = output_file.replace('.nii', f'_{series_description}.nii')
            nib.save(nifti_img, output_file_series)
            print(f"Saved NIfTI file for series '{series_description}' to: {output_file_series}")
        else:
            print(f"Skipping series '{series_description}' due to insufficient slices.")
    
    if not series_data:
        raise ValueError(f'No valid DICOM data found in {input_folder}.')

# Update affine matrix with rotation
def create_rotation_matrix(degrees):
    """ Create a rotation matrix for a given angle in degrees. """
    radians = np.deg2rad(degrees)
    cos_angle = np.cos(radians)
    sin_angle = np.sin(radians)

    return np.array([
        [cos_angle, -sin_angle, 0, 0],
        [sin_angle, cos_angle, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def update_affine_with_rotation(affine, degrees):
    """ Update affine matrix with a rotation. """
    rotation_matrix = create_rotation_matrix(degrees)
    return affine @ rotation_matrix

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
