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
    Load valid DICOM files in the input_dir, group by SeriesDescription,
    sort slices by InstanceNumber, apply soft tissue windowing, 
    and save each series as a 3D NRRD file in the output_dir.
    """
    print(f"Processing folder: {input_dir}")
    valid_dicom_files = validate_dicom_files(input_dir)

    if not valid_dicom_files:
        print(f"No valid DICOM files found in {input_dir}. Skipping...")
        return count

    print(f"Found {len(valid_dicom_files)} valid DICOM files.")

    # Group valid files by SeriesDescription
    series_groups = {}
    for file_path in valid_dicom_files:
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(file_path)
            reader.ReadImageInformation()

            # Get metadata
            series_description = reader.GetMetaData("0008|103e") if reader.HasMetaDataKey(
                "0008|103e") else f"Series_{count}"
            instance_number = int(reader.GetMetaData("0020|0013")) if reader.HasMetaDataKey("0020|0013") else count

            if series_description not in series_groups:
                series_groups[series_description] = []
            series_groups[series_description].append((file_path, instance_number))
        except Exception as e:
            print(f"Error reading metadata for {file_path}: {e}")

    # Process each series
    for series_description, file_list in series_groups.items():
        print(f"Processing series: {series_description} with {len(file_list)} files.")

        # Sort files by InstanceNumber
        sorted_files = sorted(file_list, key=lambda x: x[1] if x[1] is not None else float('inf'))
        sorted_file_paths = [file[0] for file in sorted_files]

        try:
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(sorted_file_paths)
            image = reader.Execute()
            print(image.GetSize())

            # Check for non-uniform slice spacing and resample if necessary
            if image.GetSpacing()[2] == 0:
                print("Non-uniform slice spacing detected. Attempting resampling...")
                resampler = sitk.ResampleImageFilter()
                desired_spacing = min(image.GetSpacing())  # Define your desired spacing here
                new_size = [
                    int(image.GetSize()[0] * (image.GetSpacing()[0] / desired_spacing)),
                    int(image.GetSize()[1] * (image.GetSpacing()[1] / desired_spacing)),
                    int(image.GetSize()[2] * (image.GetSpacing()[2] / desired_spacing)),
                ]
                resampler.SetOutputSpacing([image.GetSpacing()[0], image.GetSpacing()[1], desired_spacing])
                resampler.SetSize(new_size)
                resampler.SetOutputOrigin(image.GetOrigin())
                resampler.SetOutputDirection(image.GetDirection())
                resampler.SetInterpolator(sitk.sitkLinear)
                image = resampler.Execute(image)

            # Save the 3D image as NRRD
            nrrd_path = os.path.join(output_dir, f"{case_name}-{count}_images.nrrd")
            sitk.WriteImage(image, nrrd_path)
            _,_,_ = align_image(nrrd_path, flip=True)
            print(f"NRRD file saved correctly")

            count += 1
        except Exception as e:
            pass
            #print(f"Failed to convert series {series_description} to NRRD. Error: {e}")

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