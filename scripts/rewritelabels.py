import nrrd
import numpy as np
import os
"""
option = "verify_modified_labels" # Set this to "reorganize_labels" or "verify_modified_labels"

if option == "reorganize_labels":
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
    def update_nrrd_labels(nrrd_folder, label_mapping):
        for file_name in os.listdir(nrrd_folder):
            if file_name.lower().endswith('.nrrd'):
                file_path = os.path.join(nrrd_folder, file_name)
                try:
                    data, header = nrrd.read(file_path)
                    # Create a new data array where each label is replaced by its new ID
                    new_data = np.zeros_like(data)
                    for old_label, new_label in label_mapping.items():
                        new_data[data == label_mapping[old_label]] = new_label

                    # Write the updated data back to the NRRD file
                    nrrd.write(file_path, new_data, header)
                    print(f"Updated labels in {file_path}")
                except Exception as e:
                    print(f"Failed to update {file_path}: {e}")

    '''if __name__ == "__main__":
        nrrd_folder = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/multiclass_masks/arms"
        update_nrrd_labels(nrrd_folder, label_mapping)'''

elif option == "verify_modified_labels":
"""
"""
def print_unique_values_in_masks(nrrd_folder):
    print(f"Starting to scan all files in the folder: {nrrd_folder}")
    for file_name in os.listdir(nrrd_folder):
        if file_name.lower().endswith('.nrrd'):
            file_path = os.path.join(nrrd_folder, file_name)
            try:
                data, header = nrrd.read(file_path)
                unique_values = np.unique(data)
                print(f"File: {file_path} contains unique values: {unique_values}")
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")

if __name__ == "__main__":
    nrrd_folder = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/multiclass_masks/knee"
    print_unique_values_in_masks(nrrd_folder)
"""
'''

import nrrd
import numpy as np
import os


def update_labels_dynamically(nrrd_folder, label_mapping):
    print(f"Starting dynamic label update in: {nrrd_folder}")
    for file_name in os.listdir(nrrd_folder):
        if file_name.lower().endswith('.nrrd'):
            file_path = os.path.join(nrrd_folder, file_name)
            print(f"Processing file: {file_path}")
            try:
                data, header = nrrd.read(file_path)
                # Extract label information from the header
                header_labels = {int(key.split('_')[1]): value.strip() for key, value in header.items() if
                                 key.startswith('label_')}
                print(f"Labels found in header: {header_labels}")

                # Create a new data array for updated labels
                new_data = np.copy(data)
                # Map existing data labels to new labels according to header and label_mapping
                for old_label, label_name in header_labels.items():
                    if label_name in label_mapping:
                        new_label = label_mapping[label_name]
                        new_data[data == old_label] = new_label
                        print(f"Replacing {old_label} with {new_label} for {label_name}")

                # Write the updated data back to the NRRD file
                nrrd.write(file_path, new_data, header)
                print(f"Successfully updated labels in {file_path}")
            except Exception as e:
                print(f"Failed to update {file_path}: {e}")


if __name__ == "__main__":
    nrrd_folder = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/multiclass_masks/arms"
    label_mapping = {
        "Background": 0,
        "Hand_L": 9,
        "Hand_R": 10,
        "Humerus_L": 11,
        "Humerus_R": 12,
        "Radius_L": 13,
        "Radius_R": 14,
        "Ulna_L": 15,
        "Ulna_R": 16
    }
    update_labels_dynamically(nrrd_folder, label_mapping)

import nrrd
import numpy as np
import os


def remap_labels_in_nrrd(nrrd_folder):
    """
    Remaps labels in all NRRD files within `nrrd_folder` from:
       0 -> 0
       9 -> 1
       10 -> 2
       11 -> 3
       12 -> 4
       13 -> 5
       14 -> 6
       15 -> 7
       16 -> 8
    """

    # Define the remapping dictionary
    label_remap = {
        0: 0,
        9: 1,
        10: 2,
        11: 3,
        12: 4,
        13: 5,
        14: 6,
        15: 7,
        16: 8
    }

    print(f"Starting label remap in folder: {nrrd_folder}")

    for file_name in os.listdir(nrrd_folder):
        # Process only NRRD files
        if file_name.lower().endswith('.nrrd'):
            file_path = os.path.join(nrrd_folder, file_name)
            print(f"Processing file: {file_path}")
            try:
                data, header = nrrd.read(file_path)
                unique_values_before = np.unique(data)

                # Create a new data array for updated labels
                new_data = np.copy(data)

                # Remap each old label to its new label
                for old_val, new_val in label_remap.items():
                    new_data[data == old_val] = new_val

                # Optionally, you might want to clear/ update label_x entries in the header
                # to reflect the new labels (this is optional and depends on your workflow).
                # For example:
                #   for key in list(header.keys()):
                #       if key.startswith("label_"):
                #           del header[key]
                # Then re-inject new labels as needed.

                # Write the updated data back
                nrrd.write(file_path, new_data, header)

                unique_values_after = np.unique(new_data)
                print(f"File: {file_name}\n"
                      f"  Before remap: {unique_values_before}\n"
                      f"  After remap:  {unique_values_after}")
            except Exception as e:
                print(f"Failed to update {file_path}: {e}")


if __name__ == "__main__":
    # Replace with your actual path to the folder containing the multiclass NRRD files
    nrrd_folder = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/multiclass_masks/armsx "
    remap_labels_in_nrrd(nrrd_folder)
'''
import os
import nrrd
import numpy as np


def binarize_labels_in_nrrd(nrrd_folder):
    """
    For each NRRD file in nrrd_folder:
       - Keeps label 0 (background) as 0
       - Converts all other labels to 1
    """
    print(f"Starting binarization in folder: {nrrd_folder}")

    for file_name in os.listdir(nrrd_folder):
        if file_name.lower().endswith('.nrrd'):
            file_path = os.path.join(nrrd_folder, file_name)
            print(f"Processing file: {file_path}")
            try:
                data, header = nrrd.read(file_path)

                # Show unique labels before binarizing
                unique_values_before = np.unique(data)

                # Create a new data array where:
                # background (0) stays 0, and everything else becomes 1
                new_data = np.where(data == 0, 0, 1)


                # Save the updated array back to disk
                nrrd.write(file_path, new_data, header)

                # Show unique labels after binarizing
                unique_values_after = np.unique(new_data)
                print(f"  Before: {unique_values_before}")
                print(f"  After:  {unique_values_after}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")


if __name__ == "__main__":
    # Replace with your actual folder path
    nrrd_folder = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/multiclass_masks/binary"
    binarize_labels_in_nrrd(nrrd_folder)
cases = ['240033-2', '240047','240002-2', '240045-2', '240047-2', '240023-2-2']