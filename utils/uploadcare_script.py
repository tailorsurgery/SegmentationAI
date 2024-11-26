from pyuploadcare import Uploadcare
import re
import os

option = 'upload'

# Function to delete files starting with '666F'
def delete_files_with_prefix(prefix):
    # Get a list of all files (only stored, not removed files)
    files = uc.list_files(stored=True)

    # Regex to match filenames that start with the prefix
    prefix_pattern = re.compile(prefix)

    # Iterate over the files and check if they match the prefix
    for file in files:
        if prefix_pattern.match(file.filename):
            print(f"Deleting file: {file.filename}")
            file.delete()  # Delete the file

def upload_nii_files(directory, tag):
    """Upload .nii files from the specified directory to Uploadcare."""
    for filename in os.listdir(directory):
        if filename.endswith(".nii"):
            print(f"Processing file: {filename}")
            file_path = os.path.join(directory, filename).replace('\\', '/')  # Normalize the path
            
            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"File does not exist: {file_path}")
                continue
            
            print(f"Preparing to upload: {file_path}")

            try:
                # Upload the file to Uploadcare
                print(f"Uploading {filename}...")
                with open(file_path, 'rb') as file_object:
                    ucare_file = uc.upload(file_object)

                # Add a tag to the uploaded file
                ucare_file.tag(tag)  # Uncommented to tag the uploaded file
                print(f"Successfully uploaded {filename} with tag '{tag}'.")

            except Exception as e:
                print(f"Error uploading {filename}: {e}")

if option == 'delete':
    # Initialize Uploadcare client with your public and secret keys
    PUBLIC_KEY = 'aaa25a16a8fc34e2edc3'
    SECRET_KEY = 'f8d49ff24d84eee7af4d'
    uc = Uploadcare(PUBLIC_KEY, SECRET_KEY)
    delete_files_with_prefix('Hacer')

elif option == 'upload':
    # Set up your Uploadcare API credentials
    PUBLIC_KEY = '5356fe74c76435df8e3f'
    SECRET_KEY = 'ac7dc7db3f2bd6a5f08f'
    uc = Uploadcare(PUBLIC_KEY, SECRET_KEY)

    # Define the directories containing the NIfTI files
    nii_image_directory = r"C:/Users/samya/Downloads/RESEGMENTED/dataset/images"
    nii_mask_directory = r"C:\\Users\\samya\\Downloads\\RESEGMENTED\\dataset\\masks"

    # Define tags for images and masks
    image_tag_name = "images"
    mask_tag_name = "masks"

    # Upload files from both directories
    print("Uploading NIfTI images...")
    upload_nii_files(nii_image_directory, image_tag_name)

    print("Uploading NIfTI masks...")
    # Uncomment to upload masks
    # upload_nii_files(nii_mask_directory, mask_tag_name) 

    print("All uploads completed.")
