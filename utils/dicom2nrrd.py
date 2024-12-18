import os
import SimpleITK as sitk
import numpy as np


from utils.align_image_multiclass_mask import *


def validate_dicom_files(input_dir):
    """
    Validate DICOM files in the directory and return a list of valid file paths.
    """
    valid_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Validate file using SimpleITK
                sitk.ReadImage(file_path)
                valid_files.append(file_path)
            except Exception:
                print(f"Skipping non-DICOM file: {file_path}")
    return valid_files


def apply_windowing(image, window_level=40, window_width=400):
    """
    Apply windowing to the image using the specified window level and width.
    """
    min_intensity = window_level - (window_width / 2)
    max_intensity = window_level + (window_width / 2)
    windowed_image = sitk.IntensityWindowing(
        image,
        windowMinimum=min_intensity,
        windowMaximum=max_intensity,
        outputMinimum=0.0,
        outputMaximum=255.0,
    )
    return windowed_image

def load_and_convert_dicom_to_nrrd(input_dir, output_dir, case_name, count):
    """
    Load valid DICOM files, group by SeriesNumber, sort groups by a header,
    and save each group as a 3D NRRD file.
    """
    print(f"Processing folder: {input_dir}")
    valid_dicom_files = validate_dicom_files(input_dir)

    if not valid_dicom_files:
        print(f"No valid DICOM files found in {input_dir}. Skipping...")
        return count

    print(f"Found {len(valid_dicom_files)} valid DICOM files.")

    # Group files by SeriesNumber
    series_groups = {}
    group_metadata = {}  # To store additional headers for sorting
    for file_path in valid_dicom_files:
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(file_path)
            reader.ReadImageInformation()

            # Extract SeriesNumber and sorting metadata
            series_number = int(reader.GetMetaData("0020|0011")) if reader.HasMetaDataKey("0020|0011") else -1
            instance_number = int(reader.GetMetaData("0020|0013")) if reader.HasMetaDataKey("0020|0013") else 0

            # Extract the header to sort groups later
            study_date = reader.GetMetaData("0008|0020") if reader.HasMetaDataKey("0008|0020") else ""

            if series_number not in series_groups:
                series_groups[series_number] = []
                group_metadata[series_number] = study_date

            series_groups[series_number].append((file_path, instance_number))
        except Exception as e:
            print(f"Error reading metadata for {file_path}: {e}")

    # Sort the groups by the chosen header
    sorted_series_numbers = sorted(series_groups.keys(), key=lambda sn: group_metadata.get(sn, ""))

    # Process each sorted group
    for series_number in sorted_series_numbers:
        file_list = series_groups[series_number]
        print(f"Processing SeriesNumber: {series_number} with {len(file_list)} files. Sort key: {group_metadata[series_number]}")

        # Sort files by InstanceNumber within the group
        sorted_files = sorted(file_list, key=lambda x: x[1])
        sorted_file_paths = [file[0] for file in sorted_files]

        try:
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(sorted_file_paths)
            image = reader.Execute()
            print(f"Image size: {image.GetSize()}")

            # Save the 3D image as NRRD
            nrrd_path = os.path.join(output_dir, f"{case_name}-{count}_Series_{series_number}.nrrd")
            sitk.WriteImage(image, nrrd_path)
            print(f"NRRD file saved: {nrrd_path}")

            count += 1
        except Exception as e:
            print(f"Failed to process SeriesNumber {series_number}. Error: {e}")

    return count

def process_images(input_dir, output_dir, case_number):
    """
    Process all subdirectories in the input_dir, convert DICOMs to NRRD,
    and save them in the output_dir.
    """
    count = 1
    for folder in os.listdir(input_dir):
        case_path = os.path.join(input_dir, folder)
        if not os.path.isdir(case_path):
            continue
        print(f"Processing folder: {folder}")
        try:
            count = load_and_convert_dicom_to_nrrd(case_path, output_dir, case_number, count)
        except Exception as e:
            print(f"Failed to process folder {folder}: {e}")
    print("All cases processed successfully.")

def multiclass_mask_to_stl(mask_path, output_dir):
    """
    Convert a multiclass mask to a 3D STL file for visualization.
    """
    # Load the multiclass mask
    multiclass_mask = load_multiclass_mask(mask_path)

    # Convert the mask to a binary mask
    binary_mask = np.zeros_like(multiclass_mask, dtype=np.uint8)
    binary_mask[multiclass_mask > 0] = 255
