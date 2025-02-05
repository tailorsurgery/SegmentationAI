import os
import pydicom
import nibabel as nib
import numpy as np

def dicom2nifti_grouped(dicom_files, series_description, elim_grad=False, elim_slc=False):
    all_pixel_data = []

    # Sort DICOM files by Instance Number
    dicom_files.sort(key=lambda x: int(pydicom.dcmread(x).InstanceNumber) if hasattr(pydicom.dcmread(x), 'InstanceNumber') else float('inf'))

    for dicom_file in dicom_files:
        dcm_hdr = pydicom.dcmread(dicom_file)

        # Check if pixel data exists
        if not hasattr(dcm_hdr, 'PixelData'):
            print(f"No pixel data found in file: {dicom_file}")
            continue  # Skip processing this file

        # Extract pixel data
        pixel_data = dcm_hdr.pixel_array.astype(np.float32)

        # Handle rescale slope and intercept
        rescale_slope = float(getattr(dcm_hdr, 'RescaleSlope', 1.0))  # Default to 1
        rescale_intercept = float(getattr(dcm_hdr, 'RescaleIntercept', 0.0))  # Default to 0

        # Apply scaling to pixel data
        pixel_data = pixel_data * rescale_slope + rescale_intercept
        print(f"Pixel data shape: {pixel_data.shape} from file: {dicom_file}")
        all_pixel_data.append(pixel_data)

    if all_pixel_data:
        # Stack pixel data along the first axis (slices)
        try:
            all_pixel_data = np.stack(all_pixel_data, axis=0)
        except ValueError as e:
            print(f"Error stacking pixel data: {e}")
            return

        # Save as NIfTI
        # Sanitize series_description for valid filename
        sanitized_series_description = ''.join(c for c in series_description if c.isalnum() or c in (' ', '_')).rstrip()
        nifti_file = f"{sanitized_series_description}.nii"
        nifti_image = nib.Nifti1Image(all_pixel_data, affine=np.eye(4))
        nib.save(nifti_image, nifti_file)
        print(f"NIfTI saved: {nifti_file}")

def main(dicom_directory, elim_grad=False, elim_slc=False):
    series_dict = {}

    # Load all DICOM files from the directory
    for root, dirs, files in os.walk(dicom_directory):
        for file in files:
            if file.endswith('.dcm'):
                dicom_file = os.path.join(root, file)
                dcm_hdr = pydicom.dcmread(dicom_file)

                # Debug output for the file being processed
                print(f"Processing DICOM file: {dicom_file}")
                # Use Series Description as the key for grouping
                series_description = getattr(dcm_hdr, 'SeriesDescription', "Unknown_Series")
                
                if series_description not in series_dict:
                    series_dict[series_description] = []
                
                series_dict[series_description].append(dicom_file)

    print(f"Series groups: {list(series_dict.keys())}")

    # Process each series group
    for series_description, dicom_files in series_dict.items():
        dicom2nifti_grouped(dicom_files, series_description, elim_grad=elim_grad, elim_slc=elim_slc)

# Example usage
case = '240002'
dicom_directory = f"/Users/samyakarzazielbachiri/Documents/SegmentationAI/scripts/input/81409_Dra.Marian Vives_Hospital Clinic"
main(dicom_directory)
