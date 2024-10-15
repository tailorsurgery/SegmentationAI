import nibabel as nib

# Load the anonymized NIfTI file
anonymized_file_path = r"C:\Users\samya\Downloads\RESEGMENTED\dataset\images\240002_images_anonymized.nii"
anonymized_nii_image = nib.load(anonymized_file_path)

# Print the header info
print(f"Anonymized header info for {anonymized_file_path}:")
print(anonymized_nii_image.header)
