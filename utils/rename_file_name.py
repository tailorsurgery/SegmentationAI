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
import os

def rename_stl_files(directory, case_number):
    """Change name to casenumber_mask_index.stl"""
    mask_index = 1  # Start from 1 for clarity in filenames
    for filename in os.listdir(directory):
        if filename.endswith(".stl"):
            new_filename = f'{case_number}_mask_{mask_index}.stl'
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)

            # Check if the new filename already exists to avoid overwriting
            if os.path.exists(new_file_path):
                print(f"Warning: {new_file_path} already exists and may be overwritten.")
                continue  # Skip renaming to prevent overwriting

            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {old_file_path} to {new_file_path}')
            mask_index += 1

def create_cases_list(directory):
    """Create a list of case numbers from the directory names."""
    case_numbers = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path):
            case_numbers.append(item)
    return case_numbers

# Base path containing case directories
path = r'/Users/samyakarzazielbachiri/Documents/TS_DATASET/dataset/binary_stls/'

# Get list of case numbers
case_numbers = create_cases_list(path)
print(f"Case numbers: {case_numbers}")
#case_numbers = ['240014']
# Iterate through each case and rename .stl files
for case in case_numbers:
    target_directory = os.path.join(path, case)
    if not os.path.exists(target_directory):
        print(f"Directory not found: {target_directory}")
        continue  # Skip if the directory does not exist
    
    print(f"Renaming files in directory: {target_directory}")
    rename_stl_files(target_directory, case)
