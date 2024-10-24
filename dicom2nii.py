from pathlib import Path
import dicom2nifti
import nibabel as nib
import os
import pydicom

# Define paths
dicom_directory = Path('./240084')
output_dir = Path('./scripts/temp_nifti')
output_file = output_dir / 'nifti_file.nii'

# Check if DICOM directory contains files
if not dicom_directory.exists() or not any(dicom_directory.glob('*.dcm')):
    print(f"No valid DICOM files found in {dicom_directory}.")
else:
    # List DICOM files
    print("DICOM files found:", list(dicom_directory.glob('*.dcm')))

    try:
        # Convert DICOM directory to NIfTI file
        dicom2nifti.convert_directory(dicom_directory, str(output_dir), compression=True, reorient=True)

        # Check if the NIfTI file was created
        if output_file.exists():
            # Load the NIfTI file
            nifti_image = nib.load(str(output_file))
            print(nifti_image.shape)
            print(nifti_image.header)
        else:
            print(f"NIfTI file was not created at {output_file}.")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")
