import os
import SimpleITK as sitk


# Function to load and convert DICOM to NRRD
def load_and_convert_dicom_to_nrrd(input_dir, output_dir, case_name, count):
    """
    Load all DICOM series in the input_dir and save each as a separate NRRD file in the output_dir.
    """
    print("Loading and converting DICOM images to NRRD...")
    reader = sitk.ImageSeriesReader()

    # Get all series in the folder
    series_ids = reader.GetGDCMSeriesIDs(input_dir)
    if not series_ids:
        raise ValueError(f"No DICOM series found in directory: {input_dir}")

    print(f"Found {len(series_ids)} series in {input_dir}.")

    # Loop through all series
    for series_id in series_ids:
        print(f"Processing series: {series_id}")

        # Get file names for the current series
        dicom_files = reader.GetGDCMSeriesFileNames(input_dir, series_id)
        reader.SetFileNames(dicom_files)
        image = reader.Execute()

        # Save each series as an NRRD file
        nrrd_path = os.path.join(output_dir, f"{case_name}-{count}.nrrd")
        sitk.WriteImage(image, nrrd_path)
        print(f"NRRD file saved at: {nrrd_path}")
        count += 1  # Increment count for each series processed

    return count
