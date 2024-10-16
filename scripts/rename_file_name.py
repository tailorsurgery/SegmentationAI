import os

def append_suffix_to_nii_files(directory, suffix):
    """Append a suffix to all .nii files in the specified directory."""
    # Iterate through all files in the given directory
    for filename in os.listdir(directory):
        # Check if the file ends with .nii
        if filename.endswith(".nii"):
            # Create the new filename by appending the suffix
            new_filename = filename.replace('.nii', f'{suffix}.nii')
            # Construct full file paths
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {old_file_path} to {new_file_path}')

# Define the target directory and suffix
target_directory = r'C:\\Users\\samya\\Downloads\\RESEGMENTED\\dataset\\masks'
suffix = '_masks'

# Call the function
append_suffix_to_nii_files(target_directory, suffix)
