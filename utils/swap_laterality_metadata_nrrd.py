import os
import SimpleITK as sitk
import numpy as np

def swap_laterality_in_mask(mask, label_mapping):
    """
    Swap laterality of labels in the given mask.

    Args:
        mask (numpy.ndarray): The input mask as a NumPy array.
        label_mapping (dict): The label mapping dictionary.

    Returns:
        numpy.ndarray: Mask with swapped laterality.
    """
    swapped_mask = mask.copy()

    # Define left-to-right and right-to-left mappings
    laterality_map = {
        label_mapping[f"{bone}_L"]: label_mapping[f"{bone}_R"]
        for bone in [
            "Femur", "Fibula", "Patella", "Tibia", "Hand", "Humerus", "Radius", "Ulna"
        ]
    }
    laterality_map.update({v: k for k, v in laterality_map.items()})  # Add reverse mapping

    # Swap the labels in the mask
    for left_label, right_label in laterality_map.items():
        swapped_mask[mask == left_label] = right_label

    return swapped_mask

if __name__ == "__main__":
    nrrd_folder = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/multiclass_masks/arms_mirrored"
    label_mapping = {
        "Background": 0,
        "Femur_L": 1,
        "Femur_R": 2,
        "Fibula_L": 3,
        "Fibula_R": 4,
        "Patella_L": 5,
        "Patella_R": 6,
        "Tibia_L": 7,
        "Tibia_R": 8,
        "Hand_L": 9,
        "Hand_R": 10,
        "Humerus_L": 11,
        "Humerus_R": 12,
        "Radius_L": 13,
        "Radius_R": 14,
        "Ulna_L": 15,
        "Ulna_R": 16
    }

    # Process each .nrrd file in the folder
    for filename in os.listdir(nrrd_folder):
        if filename.endswith(".nrrd"):
            mask_path = os.path.join(nrrd_folder, filename)

            print(f"Processing: {filename}")

            # Load the mask
            mask_image = sitk.ReadImage(mask_path)
            mask_array = sitk.GetArrayFromImage(mask_image)

            # Swap laterality
            swapped_mask_array = swap_laterality_in_mask(mask_array, label_mapping)

            # Save the updated mask
            swapped_mask_image = sitk.GetImageFromArray(swapped_mask_array)
            swapped_mask_image.CopyInformation(mask_image)

            output_path = os.path.join(nrrd_folder, f"{filename}")
            sitk.WriteImage(swapped_mask_image, output_path)

            print(f"Saved swapped mask to: {output_path}")
