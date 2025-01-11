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
    update_labels_dynamically(nrrd_folder, label_mapping)