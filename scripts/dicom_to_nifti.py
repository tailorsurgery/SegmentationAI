import os
import numpy as np
import nibabel as nib
import pydicom
import time
from skimage.transform import resize
from check import check_nii_file

'''case = 240068  # TODO: Update case number
option = 'image'  # 'image' or 'mask'''

def is_dicom_file(file_path):
    try:
        pydicom.dcmread(file_path)
        return True
    except Exception:
        return False

def load_dicom_folder(folder_path, target_shape=(512, 512)):
    dicom_files = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if is_dicom_file(file_path):
            dicom_files.append(file_path)

    if not dicom_files:
        print(f'Warning: No DICOM files found in {folder_path}.')
        return None

    dicom_files.sort(key=lambda x: int(pydicom.dcmread(x).InstanceNumber))

    num_slices = len(dicom_files)
    shape = (num_slices, target_shape[0], target_shape[1])
    data = np.zeros(shape, dtype=np.float32)

    for i, dicom_file in enumerate(dicom_files):
        try:
            dicom = pydicom.dcmread(dicom_file)
            if 'PixelData' in dicom:
                if hasattr(dicom, 'SamplesPerPixel') and dicom.SamplesPerPixel > 1:
                    rgb_image = dicom.pixel_array
                    grayscale_image = np.dot(rgb_image[...,:3], [0.2989, 0.5870, 0.1140])
                    resized_image = resize(grayscale_image, target_shape, mode='reflect', anti_aliasing=True)
                    data[i, :, :] = resized_image
                else:
                    resized_image = resize(dicom.pixel_array, target_shape, mode='reflect', anti_aliasing=True)
                    data[i, :, :] = resized_image
                print(f'Processed {dicom_file}')
            else:
                print(f'Warning: No pixel data in {dicom_file}.')
        except Exception as e:
            print(f'Error processing {dicom_file}: {e}')

    return data

def dicom_to_nifti(input_folder, output_file):
    all_data = []
    
    for root, dirs, files in os.walk(input_folder):
        dicom_files = [f for f in files if is_dicom_file(os.path.join(root, f))]
        if dicom_files:
            print(f'Processing folder: {root}')
            data = load_dicom_folder(root)
            if data is not None:
                all_data.append(data)
        else:
            print(f'No DICOM files found in {root}.')

    if not all_data:
        raise ValueError(f'No DICOM data found in any of the subdirectories of {input_folder}.')

    combined_data = np.concatenate(all_data, axis=0)
    nifti_img = nib.Nifti1Image(combined_data, np.eye(4))
    nib.save(nifti_img, output_file)
    print(f'Successfully saved NIfTI file to: {output_file}')


# Main processing section
def main_function(case, option):
    print(f"****Processing case {case}...")
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
for case in ['240008']:   
    main_function(case, 'image')