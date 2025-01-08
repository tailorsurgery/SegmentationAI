import SimpleITK as sitk
import numpy as np

def delete_class_and_update_labels(nrrd_path, class_to_delete, updated_labels, output_path=None):
    """
    Delete a specific class (mask) from a multiclass NRRD file and update the header with new labels.

    Args:
        nrrd_path (str): Path to the input NRRD file.
        class_to_delete (int): The class index to delete.
        updated_labels (dict): Dictionary mapping class indices to new names.
        output_path (str): Path to save the updated NRRD file (optional, overwrites if None).

    Returns:
        None
    """
    # Load the NRRD file
    print(f"Loading NRRD file: {nrrd_path}")
    mask_image = sitk.ReadImage(nrrd_path)
    mask_array = sitk.GetArrayFromImage(mask_image)

    # Remove the specified class
    print(f"Deleting class {class_to_delete} from the mask...")
    mask_array[mask_array == class_to_delete] = 0

    # Remove metadata for the deleted class
    metadata_keys = mask_image.GetMetaDataKeys()
    for key in metadata_keys:
        if key.startswith(f"label_{class_to_delete}"):
            print(f"Removing metadata: {key}")
            mask_image.EraseMetaData(key)

    # Add updated labels to the metadata
    print("Updating metadata with new labels...")
    for class_index, class_name in updated_labels.items():
        print(f"Adding metadata: label_{class_index} := {class_name}")
        mask_image.SetMetaData(f"label_{class_index}", class_name)

    # Convert back to SimpleITK image
    updated_mask_image = sitk.GetImageFromArray(mask_array)
    updated_mask_image.CopyInformation(mask_image)

    # Save the updated NRRD file
    if output_path is None:
        output_path = nrrd_path  # Overwrite the original file
    sitk.WriteImage(updated_mask_image, output_path)
    print(f"Updated NRRD saved at: {output_path}")

# Example Usage
path = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset"
nrrd_path = path + "/multiclass_masks/arms/240050-1_multiclass_mask.nrrd"
output_path = path + "/multiclass_masks/arms/240050-1_multiclass_mask.nrrd"
class_to_delete = 1  # Example: Delete class with label 2
updated_labels = {
    0: "Background",
    1: "Humerus_R"
}
delete_class_and_update_labels(nrrd_path, class_to_delete, updated_labels, output_path)
