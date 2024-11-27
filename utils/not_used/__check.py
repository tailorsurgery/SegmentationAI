import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import time

def check_nii_file(case, option):
    print(f"****Checking case {case}...")

    if option == 'mask':
        # Load the NIfTI file for masks
        nifti_file_path = f'./../TS_DATASET/dataset/images/{case}_masks.nii'  # Update this path
        nifti_img = nib.load(nifti_file_path)
        mask_data = nifti_img.get_fdata()

        # Debugging print
        print(f"Mask data shape: {mask_data.shape}")

        # Choose a mask index (e.g., 0 for the first mask)
        mask_index = 0

        # Display a middle slice of the 3D mask
        slice_index = mask_data.shape[0] // 2  # Middle slice along the z-axis
        plt.imshow(mask_data[slice_index, :, :], cmap='gray')
        plt.title(f'Mask {mask_index + 1} - Slice {slice_index}')
        plt.axis('off')
        plt.show()

    elif option == 'image':
        # Load the NIfTI file for images
        nifti_file = f'./../TS_DATASET/dataset/images/{case}_images.nii'
        nifti_img = nib.load(nifti_file)

        # Get the image data
        img_data = nifti_img.get_fdata(dtype=np.float32)

        # Debugging prints
        print("Data Shape:", img_data.shape)

        # Visualize the middle slice
        middle_slice_index = img_data.shape[0] // 2
        print(f"Middle Slice Index: {middle_slice_index}")

        # Ensure that the data isn't empty
        if img_data.size > 0:
            plt.imshow(img_data[middle_slice_index, :, :], cmap='gray')
            plt.title(f'Middle Slice (Index {middle_slice_index})')
            plt.axis('off')
            plt.show()
        else:
            print("Warning: No image data found.")

        # Print the unique values in the image data (sample only first 10)
        unique_values = np.unique(img_data)
        print(f"Unique values in the image data (sample): {unique_values[:10]}")

# Call the function for case '240044' and 'images'

# Optional: Pause the script to ensure plot stays on screen for a bit
time.sleep(5)


'''import pydicom
import os
import time
def print_dicom_info(file_path):
    try:
        dicom = pydicom.dcmread(file_path)
        print(f'DICOM file: {file_path}')
        print(f'Study Instance UID: {dicom.StudyInstanceUID}')
        print(f'Series Instance UID: {dicom.SeriesInstanceUID}')
        print(f'Patient ID: {dicom.PatientID}')
        print(f'Patient Name: {dicom.PatientName}')
        print(f'Study Description: {dicom.StudyDescription}')
        print(f'Series Description: {dicom.SeriesDescription}')
        print(f'Instance Number: {dicom.InstanceNumber}')
        print(f'Samples Per Pixel: {dicom.SamplesPerPixel}')
        print(f'Photometric Interpretation: {dicom.PhotometricInterpretation}')
        print(f'Pixel Data: {dicom.PixelData is not None}')
        print('-' * 40)
    except Exception as e:
        print(f'Error reading {file_path}: {e}')
def is_dicom_file(file_path):
    try:
        pydicom.dcmread(file_path, force=True)  # Use force=True to read non-standard DICOM files
        return True
    except (pydicom.errors.InvalidDicomError, FileNotFoundError):
        return False
# Function to print DICOM info for all files in a folder
def print_dicom_folder_info(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if is_dicom_file(file_path):
            print_dicom_info(file_path)
            time.sleep(5)

# Example usage
folder_path = './240084'  # Change this to your DICOM folder path
print_dicom_folder_info(folder_path)
'''