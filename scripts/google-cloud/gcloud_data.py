# download_gcloud_data.py

from google.cloud import storage
import os


def download_from_gcs(bucket_name, prefix, local_dir):
    """
    Downloads files from a Google Cloud Storage bucket.

    Args:
        bucket_name (str): Name of the GCS bucket.
        prefix (str): Prefix (directory path) within the GCS bucket.
        local_dir (str): Local directory to save the downloaded files.
    """
    # Initialize GCS client
    client = storage.Client()

    # Access the bucket
    bucket = client.get_bucket(bucket_name)

    # List all files with the specified prefix
    blobs = bucket.list_blobs(prefix=prefix)

    # Download each file
    for blob in blobs:
        # Skip "directory-like" blobs (end with '/')
        if blob.name.endswith('/'):
            continue

        # Generate local file path
        local_path = os.path.join(local_dir, os.path.relpath(blob.name, prefix))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"Downloading {blob.name} to {local_path}...")
        blob.download_to_filename(local_path)
    print("Download complete.")


if __name__ == "__main__":
    # GCS bucket and prefix information
    BUCKET_NAME = "segmentai_dataset"
    IMAGE_PREFIX = "images/"
    MASK_PREFIX = "multiclass_masks/"

    # Get current working directory
    path = "C:/Users/Laura Montserrat/Documents/Samya/SegmentationAI/data/"

    # Local directories for saving the data
    LOCAL_IMAGE_DIR = os.path.join(path, "segmentai_dataset/images")
    LOCAL_MASK_DIR = os.path.join(path, "segmentai_dataset/multiclass_masks")

    print(f"Local image directory: {LOCAL_IMAGE_DIR}")
    print(f"Local mask directory: {LOCAL_MASK_DIR}")

    # Download images and masks
    download_from_gcs(BUCKET_NAME, IMAGE_PREFIX, LOCAL_IMAGE_DIR)
    download_from_gcs(BUCKET_NAME, MASK_PREFIX, LOCAL_MASK_DIR)
