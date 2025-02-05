import os
import numpy as np
import nibabel as nib
import pydicom
import time


def is_dicom_file(file_path):
    """
    Check if a file is a valid DICOM file.
    """
    try:
        pydicom.dcmread(file_path)
        return True
    except Exception:
        return False


def reslice_to_consistent_shape(slice_image, target_shape):
    """
    Reslice the DICOM slice to match the target shape by padding or cropping.
    """
    slice_height, slice_width = slice_image.shape
    target_height, target_width = target_shape

    # Create a blank array for the target shape
    padded_slice = np.zeros((target_height, target_width), dtype=slice_image.dtype)

    # Calculate offsets for centering the slice
    y_offset = max(0, (target_height - slice_height) // 2)
    x_offset = max(0, (target_width - slice_width) // 2)

    # Insert slice into the padded array
    padded_slice[y_offset:y_offset + slice_height, x_offset:x_offset + slice_width] = slice_image[
        :min(slice_height, target_height), :min(slice_width, target_width)
    ]
    return padded_slice


def load_dicom_folder(dicom_files):
    """
    Load and stack DICOM slices into a 3D numpy array.
    """
    dicom_with_instance_numbers = []

    # Read DICOM files and sort by InstanceNumber
    for dicom_file in dicom_files:
        try:
            dicom = pydicom.dcmread(dicom_file)
            instance_number = getattr(dicom, 'InstanceNumber', None)
            if instance_number is not None:
                dicom_with_instance_numbers.append((dicom_file, int(instance_number)))
        except Exception as e:
            print(f"Error reading DICOM file {dicom_file}: {e}")

    dicom_with_instance_numbers.sort(key=lambda x: x[1])

    slices = []
    for dicom_file, _ in dicom_with_instance_numbers:
        try:
            dicom = pydicom.dcmread(dicom_file)
            if 'PixelData' in dicom:
                slices.append(dicom.pixel_array)
        except Exception as e:
            print(f"Error processing {dicom_file}: {e}")

    if not slices:
        return None

    # Determine the target shape based on the largest slice
    target_shape = (max(s.shape[0] for s in slices), max(s.shape[1] for s in slices))

    # Reslice and stack slices into a 3D array
    stacked_data = np.stack([reslice_to_consistent_shape(slice_image, target_shape) for slice_image in slices], axis=0)
    return stacked_data


def dicom_to_nifti(input_folder, output_file):
    """
    Convert all DICOM series in a folder into separate NIfTI files.
    """
    series_data = {}

    # Collect DICOM files by SeriesDescription
    for root, dirs, files in os.walk(input_folder):
        dicom_files = [os.path.join(root, f) for f in files if is_dicom_file(os.path.join(root, f))]
        if dicom_files:
            print(f'Processing folder: {root}')

            for dicom_file in dicom_files:
                try:
                    dicom = pydicom.dcmread(dicom_file)
                    series_description = getattr(dicom, 'SeriesDescription', 'Unknown Series')

                    if series_description not in series_data:
                        series_data[series_description] = []
                    series_data[series_description].append(dicom_file)
                except Exception as e:
                    print(f"Error reading DICOM file {dicom_file}: {e}")

    # Process each series separately and save as NIfTI
    for series_description, dicom_files in series_data.items():
        print(f"Processing series: {series_description}")

        try:
            data = load_dicom_folder(dicom_files)

            if data is not None and data.ndim == 3 and data.shape[0] > 1:  # Check for valid 3D shape
                nifti_img = nib.Nifti1Image(data, np.eye(4))
                output_file_series = output_file.replace('.nii', f'_{series_description}.nii')
                nib.save(nifti_img, output_file_series)
                print(f"Saved NIfTI file for series '{series_description}' to: {output_file_series}")
            else:
                print(f"Skipping series '{series_description}' due to insufficient slices or invalid shape.")
        except Exception as e:
            print(f"Error processing series '{series_description}': {e}")

    if not series_data:
        raise ValueError(f'No valid DICOM data found in {input_folder}.')


def main_function(case, option):
    """
    Main entry point for the script.
    """
    input_path = f'/Users/samyakarzazielbachiri/Documents/SegmentationAI/scripts/input'
    output_path = f'/Users/samyakarzazielbachiri/Documents/SegmentationAI/scripts/output/{case}_images.nii'

    try:
        time_start = time.time()
        dicom_to_nifti(input_path, output_path)
        time_end = time.time()
        print(f"Total time taken: {time_end - time_start:.2f} seconds")
    except Exception as e:
        print(f"Error processing case {case}: {e}")


if __name__ == "__main__":
    main_function('240051', 'image')