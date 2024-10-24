from google.cloud import storage
import nibabel as nib
import numpy as np
import os

def connect_to_bucket(bucket_name):
    """Establishes connection to a Google Cloud Storage bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    print(f"Connected to bucket {bucket_name}")
    return bucket

def download_nifti(bucket_name, blob_name, local_path):
    """Downloads a NIfTI file from Google Cloud Storage."""
    bucket = connect_to_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Download the NIfTI file to the local system
    blob.download_to_filename(local_path)
    print(f"Downloaded {blob_name} to {local_path}")

def upload_nifti(bucket_name, local_file_path, destination_blob_name):
    """Uploads a processed NIfTI file back to Google Cloud Storage."""
    bucket = connect_to_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    # Upload the NIfTI file to the bucket
    blob.upload_from_filename(local_file_path)
    print(f"Uploaded {local_file_path} to {destination_blob_name}")

# Example usage
case = "240002"
bucket_name = "segmentai_dataset"  # Replace with your bucket name
nifti_images_blob_name = f"images/{case}_images.nii"  # NIfTI file in the bucket
nifti_masks_blob_name = f"masks/{case}_masks.nii"  # NIfTI file in the bucket
local_nifti_image_path = f"./temp_nifti/images/{case}_images.nii" # Local path to save the processed images
local_nifti_masks_path = f"./temp_nifti/masks/{case}_masks.nii"  # Local path to save the processed masks

# Download the NIfTI file from the bucket
download_nifti(bucket_name, nifti_images_blob_name, local_nifti_image_path)
upload_nifti(bucket_name, local_nifti_masks_path, nifti_masks_blob_name)