import SimpleITK as sitk
import numpy as np
import napari
import os
from PyQt5.QtWidgets import QApplication

import warnings

from tornado.gen import multi

warnings.filterwarnings("ignore", message="A QApplication is already running with 1 event loop.*")
warnings.filterwarnings("ignore", message="color_dict did not provide a default color.*")


def load_multiclass_mask(mask_path, label_mapping):
    """
    Load a multiclass mask from a file, ignoring header metadata.
    Instead, use the numeric values in the mask array and map them
    to class names via the provided label_mapping.
    """
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)

    # Identify all unique numeric labels present in the mask data
    unique_values = np.unique(mask_array)

    # Build a dictionary that maps each numeric label to a name
    labels = {}
    for val in unique_values:
        if val in label_mapping:
            labels[val] = label_mapping[val]
        else:
            # If the label_mapping doesn't define this value, assign a fallback name
            labels[val] = f"Class_{val}"

    print(f"Numeric labels found in {mask_path}: {unique_values}")
    print(f"Assigned names: {labels}")
    return mask_array, labels


def apply_windowing(image, window_level=40, window_width=400):
    """
    Apply windowing to the image using the specified window level and width.
    """
    min_intensity = window_level - (window_width / 2)
    max_intensity = window_level + (window_width / 2)
    windowed_image = sitk.IntensityWindowing(
        image,
        windowMinimum=min_intensity,
        windowMaximum=max_intensity,
        outputMinimum=0.0,
        outputMaximum=255.0,
    )
    return windowed_image


def align_image(image_path, flip=False, save=False):
    """
    Load the image and get its properties for visualization.
    Applies windowing for soft tissue visualization.
    """
    image_raw = sitk.ReadImage(image_path)
    # Apply windowing for soft tissue
    image = apply_windowing(image_raw, window_level=40, window_width=400)

    image_array = sitk.GetArrayFromImage(image)
    image_spacing = image.GetSpacing()

    if flip:
        image_array = np.flip(image_array, axis=0)
        image = sitk.GetImageFromArray(image_array)
        image.SetSpacing(image_spacing)

    if save:
        sitk.WriteImage(image, image_path)

    # We return both a NumPy spacing (for computations) and the original spacing
    return image_array, np.array(image_spacing), image_spacing


def create_color_map(multiclass_mask, labels):
    """
    Generate a unique color map for each class in the multiclass mask.
    """
    # Assign a random color to each label except 0, which we consider background
    color_map = {index: np.random.rand(3,) for index in labels if index != 0}
    color_map[0] = [1.0, 1.0, 1.0]  # Background (white)
    return color_map


def viewer_with_colored_classes(image_array, multiclass_mask, image_spacing, labels):
    """
    Create separate Napari viewers for axial, coronal, sagittal, and 3D views,
    assuming the array is shaped (Z, Y, X) and removing extra flips/rotations.
    """
    print("Loading Viewer...")

    # ---------------------------
    # 1) Generate a color map
    # ---------------------------
    color_map = create_color_map(multiclass_mask, labels)

    # ---------------------------
    # 2) Axial View
    #    - array shape: (Z, Y, X)
    #    - scale: (spacing_z, spacing_y, spacing_x)
    # ---------------------------
    axial_viewer = napari.Viewer(title="Axial View")
    axial_scale = image_spacing[::-1]  # Reverse => (sz, sy, sx)
    axial_viewer.add_image(
        image_array,
        name="Axial Image",
        colormap="gray",
        scale=axial_scale
    )

    # ---------------------------
    # 3) Coronal View
    #    - swapaxes(0,1): (Z,Y,X) => (Y,Z,X)
    #    - scale: (spacing_y, spacing_z, spacing_x)
    # ---------------------------
    coronal_viewer = napari.Viewer(title="Coronal View")
    coronal_image = np.swapaxes(image_array, 0, 1)
    coronal_scale = (image_spacing[1], image_spacing[2], image_spacing[0])
    coronal_viewer.add_image(
        coronal_image,
        name="Coronal Image",
        colormap="gray",
        scale=coronal_scale
    )

    # ---------------------------
    # 4) Sagittal View
    #    - swapaxes(0,2): (Z,Y,X) => (X,Y,Z)
    #    - scale: (spacing_x, spacing_y, spacing_z)
    # ---------------------------
    sagittal_viewer = napari.Viewer(title="Sagittal View")
    sagittal_image = np.swapaxes(image_array, 0, 2)
    sagittal_scale = (image_spacing[0], image_spacing[1], image_spacing[2])
    sagittal_viewer.add_image(
        sagittal_image,
        name="Sagittal Image",
        colormap="gray",
        scale=sagittal_scale
    )

    # ---------------------------
    # 5) 3D View
    # ---------------------------
    t3d_viewer = napari.Viewer(title="3D View", ndisplay=3)

    # ---------------------------
    # Add Mask Layers to Each Viewer
    # ---------------------------
    for cls, color in color_map.items():
        class_mask = (multiclass_mask == cls).astype(np.uint8)

        # Axial
        axial_viewer.add_labels(
            class_mask,
            name=f"Class {cls}: {labels[cls]}",
            scale=axial_scale,
            colormap={1: color},
            opacity=0.3
        )

        # Coronal
        coronal_mask = np.swapaxes(class_mask, 0, 1)
        coronal_viewer.add_labels(
            coronal_mask,
            name=f"Class {cls}: {labels[cls]}",
            scale=coronal_scale,
            colormap={1: color},
            opacity=0.5
        )

        # Sagittal
        sagittal_mask = np.swapaxes(class_mask, 0, 2)
        sagittal_viewer.add_labels(
            sagittal_mask,
            name=f"Class {cls}: {labels[cls]}",
            scale=sagittal_scale,
            colormap={1: color},
            opacity=0.5
        )

        # 3D
        t3d_viewer.add_labels(
            class_mask,
            name=f"Class {cls}: {labels[cls]}",
            scale=axial_scale,
            colormap={1: color},
            opacity=0.9
        )

    # ---------------------------
    # (Optional) Resize Windows
    # ---------------------------
    sagittal_viewer.window._qt_window.resize(1000, 100)
    coronal_viewer.window._qt_window.resize(1000, 100)
    axial_viewer.window._qt_window.resize(1000, 100)
    t3d_viewer.window._qt_window.resize(1000, 100)

    # (Optional) Positioning
    window_width = 1920
    window_height = 1059
    screen_geometry = QApplication.desktop().screenGeometry()
    screen_width = screen_geometry.width()
    screen_height = screen_geometry.height()

    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    axial_viewer.window._qt_window.move(x + 900, y)
    sagittal_viewer.window._qt_window.move(x, y + 500)
    coronal_viewer.window._qt_window.move(x, y)
    t3d_viewer.window._qt_window.move(x + 900, y + 500)

    napari.run()
def viewer_with_background(image_array, multiclass_mask, image_spacing):
    """
    Visualize only the background mask (value = 0) along with the image in Napari.
    """
    print("Loading Viewer for Background...")

    # Create a Napari viewer
    viewer = napari.Viewer(title="Background Mask Viewer")

    # Add the original image
    viewer.add_image(
        image_array,
        name="Image",
        colormap="gray",
        scale=image_spacing[::-1],  # Reverse the spacing order for (Z, Y, X)
    )

    # Extract the background mask (value = 0)
    background_mask = (multiclass_mask == 0).astype(np.uint8)

    # Add the background mask
    viewer.add_labels(
        background_mask,
        name="Background Mask",
        scale=image_spacing[::-1],
        opacity=0.5,  # Optional: Adjust opacity
        colormap={1: [1.0, 1.0, 1.0]},  # White color for the background
    )

    # Run the viewer
    napari.run()
def update_nrrd_class_names(nrrd_path, updated_names, output_path=None):
    """
    Update class names in an existing NRRD file (metadata).
    Not used in the new logic, but kept for reference if needed.
    """
    mask_image = sitk.ReadImage(nrrd_path)
    for class_value, class_name in updated_names.items():
        mask_image.SetMetaData(f"label_{class_value}", class_name)
    if output_path is None:
        output_path = nrrd_path  # Overwrite the existing file
    sitk.WriteImage(mask_image, output_path)
    print(f"Updated NRRD saved at {output_path}")


# --------------------- EXAMPLE MAIN ---------------------
if __name__ == "__main__":
    # Example: Single-case usage
    case = "240042-1"
    #case = "240025-2"
    region = "arms"
    print(f"Processing case: {case}")

    image_path = f'/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/images/{region}/{case}_images.nrrd'
    mask_path = f'/Users/samyakarzazielbachiri/Downloads/{case}_predicted_binary_mask_epoch5.nrrd'
    #mask_path = f'/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/multiclass_masks/arms/{case}_multiclass_mask.nrrd'
    # Step 1: Load image (with optional flip or save disabled for now)
    image_array, image_spacing, _ = align_image(image_path, flip=False)
    #image_array, image_spacing, _ = align_image(image_path, flip=True, save=True)

    # Step 2: Define your numeric → name mapping (not from the NRRD header!)
    label_mapping2 = {
        0: "Background",
        1: "Bones"}

    label_mapping = {
        0: "Background",
        1: "Femur_L",
        2: "Femur_R",
        3: "Fibula_L",
        4: "Fibula_R",
        5: "Patella_L",
        6: "Patella_R",
        7: "Tibia_L",
        8: "Tibia_R",
        9: "Hand_L",
        10: "Hand_R",
        11: "Humerus_L",
        12: "Humerus_R",
        13: "Radius_L",
        14: "Radius_R",
        15: "Ulna_L",
        16: "Ulna_R"
    }

    # Step 3: Load the mask array & derive labels from label_mapping (ignoring header)
    multiclass_mask, labels = load_multiclass_mask(mask_path, label_mapping2)

    # Step 4: Print how many unique classes are in the mask data
    unique_vals = np.unique(multiclass_mask)
    print(f"Unique mask values for case {case}: {unique_vals} (Count: {len(unique_vals)})")

    # Step 5: Visualize with Napari
    viewer_with_colored_classes(image_array, multiclass_mask, image_spacing, labels)
    #viewer_with_background(image_array, multiclass_mask, image_spacing)


'''
def load_multiclass_mask(mask_path): 
    """
    Load a multiclass mask from a file.
    """
    mask = sitk.ReadImage(mask_path)
    metadata_keys = mask.GetMetaDataKeys()

    # Filter out keys that start with "label_" and extract their values
    labels = {}
    for key in metadata_keys:
        if key.startswith("label_"):
            class_index = int(key.split("_")[1])  # Extract the class index
            class_name = mask.GetMetaData(key)
            labels[class_index] = class_name
    if not labels:
        print("No label_ headers found in metadata. Assigning temporary class names.")
        mask_array = sitk.GetArrayFromImage(mask)
        unique_classes = np.unique(mask_array)
        #labels = {int(cls): f"Class_{cls}" for cls in unique_classes}
        labels = {int(cls): f"Class_{cls}" for cls in unique_classes}

    print(labels)
    return sitk.GetArrayFromImage(mask), labels


def apply_windowing(image, window_level=40, window_width=400):
    """
    Apply windowing to the image using the specified window level and width.
    """
    min_intensity = window_level - (window_width / 2)
    max_intensity = window_level + (window_width / 2)
    windowed_image = sitk.IntensityWindowing(
        image,
        windowMinimum=min_intensity,
        windowMaximum=max_intensity,
        outputMinimum=0.0,
        outputMaximum=255.0,
    )
    return windowed_image

def align_image(image_path, flip=False, save=False):
    """
    Load the image and get its properties for visualization.
    """
    image_raw = sitk.ReadImage(image_path)
    # Apply windowing for soft tissue visualization
    image = apply_windowing(image_raw, window_level=40, window_width=400)

    image_array = sitk.GetArrayFromImage(image)
    image_spacing = image.GetSpacing()

    if flip:
        image_array = np.flip(image_array, axis=0)
        image = sitk.GetImageFromArray(image_array)
        #image_spacing = image.GetSpacing()
        image.SetSpacing(image_spacing)
    if save:
        sitk.WriteImage(image, image_path)

    return image_array, np.array(image_spacing), image_spacing

def create_color_map(multiclass_mask, labels):
    """
    Generate a unique color map for each class in the multiclass mask.
    """
    color_map = {index: np.random.rand(3, ) for index in labels if index != 0}  # Skip background class
    color_map[0] = [1.0, 1.0, 1.0]  # Default color (white)
    return color_map

def viewer_with_colored_classes(image_array, multiclass_mask, image_spacing, labels):
    """
    Create separate Napari viewers for axial, coronal, and sagittal views with colored classes.
    """
    print("Loading Viewer...")
    # Generate color map for the classes
    color_map = create_color_map(multiclass_mask, labels)

    # Axial view
    axial_viewer = napari.Viewer(title="Axial View")
    axial_spacing = image_spacing[[2, 1, 0]]
    axial_viewer.add_image(image_array, name="Axial Image", colormap="gray", scale=axial_spacing)

    # Coronal view
    coronal_viewer = napari.Viewer(title="Coronal View")
    coronal_image = np.swapaxes(image_array, 0, 1)
    coronal_image = np.flip(coronal_image, axis=(0, 1))
    coronal_spacing = image_spacing[[1, 2, 0]]
    coronal_viewer.add_image(coronal_image, name="Coronal Image", colormap="gray", scale=coronal_spacing)

    # Sagittal view
    sagittal_viewer = napari.Viewer(title="Sagittal View")
    sagittal_image = np.swapaxes(image_array, 0, 2)
    sagittal_image = np.rot90(sagittal_image, k=1, axes=(1, 2))
    sagittal_spacing = image_spacing[[1, 2, 0]]
    sagittal_viewer.add_image(sagittal_image, name="Sagittal Image", colormap="gray", scale=sagittal_spacing)

    # 3D view
    t3d_viewer = napari.Viewer(title="3D View", ndisplay=3)
    #t3d_viewer.add_image(image_array, name="3D Image", colormap="gray", scale=axial_spacing, rendering='mip')

    for cls, color in color_map.items():
        
        class_mask = (multiclass_mask == cls).astype(np.uint8)

        axial_viewer.add_labels(
            class_mask,
            name=f"Class {cls}: {labels[cls]}",
            scale=axial_spacing,  # Swap the axes for coronal view
            opacity=0.5,
            colormap={1: color},  # Map the binary mask to the unique color
        )

        coronal_mask = np.swapaxes(class_mask, 0, 1)
        coronal_mask = np.flip(coronal_mask, axis=(0, 1))
        coronal_viewer.add_labels(
            coronal_mask,
            name=f"Class {cls}: {labels[cls]}",
            scale=coronal_spacing,
            opacity=0.5,
            colormap={1: color},
        )


        sagittal_mask = np.swapaxes(class_mask, 0, 2)
        sagittal_mask = np.rot90(sagittal_mask, k=1, axes=(1, 2))
        sagittal_viewer.add_labels(
            sagittal_mask,
            name=f"Class {cls}: {labels[cls]}",
            scale=sagittal_spacing,
            opacity=0.5,
            colormap={1: color},
        )


        t3d_viewer.add_labels(
            class_mask,
            name=f"Class {cls}: {labels[cls]}",
            scale=axial_spacing,
            opacity=0.9,
            colormap={1: color},
        )

    # Resize the windows
    sagittal_viewer.window._qt_window.resize(1000, 100)  # Width=800, Height=600
    coronal_viewer.window._qt_window.resize(1000, 100)
    axial_viewer.window._qt_window.resize(1000, 100)
    t3d_viewer.window._qt_window.resize(1000, 100)

    window_width = 1920
    window_height = 1059
    screen_geometry = QApplication.desktop().screenGeometry()
    screen_width = screen_geometry.width()
    screen_height = screen_geometry.height()


    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    axial_viewer.window._qt_window.move(x+900, y)
    sagittal_viewer.window._qt_window.move(x, y+500)
    coronal_viewer.window._qt_window.move(x, y)
    t3d_viewer.window._qt_window.move(x+900, y+500)

    napari.run()

def update_nrrd_class_names(nrrd_path, updated_names, output_path=None):
    """
    Update class names in an existing NRRD file.

    Args:
        nrrd_path (str): Path to the existing NRRD file.
        updated_names (dict): Dictionary mapping class indices to new names.
        output_path (str): Path to save the updated file (optional, overwrites if None).
    """
    # Load the existing NRRD file
    mask_image = sitk.ReadImage(nrrd_path)

    # Update metadata with new class names
    for class_value, class_name in updated_names.items():
        mask_image.SetMetaData(f"label_{class_value}", class_name)

    # Save the updated NRRD file
    if output_path is None:
        output_path = nrrd_path  # Overwrite the existing file
    sitk.WriteImage(mask_image, output_path)
    print(f"Updated NRRD saved at {output_path}")






### Example main 1 - On single case### Example main 1 - On single case
if __name__ == "__main__":
    # Paths to image and masks
    case = '230090-2' #TODO - HEREEEEEEEE
    print(f"Processing case: {case}")
    # knee_copy = '-copy'
    knee_copy = ''
    image_path = f'/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/images/knee/{case}_images.nrrd'
    mask_path = f"/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/multiclass_masks/knee/{case}_multiclass_mask.nrrd"
    # Load image and mask
    image_array, image_spacing, _ = align_image(image_path, flip=False)

    p = 100

    if p == 1:
        #new_class_names = {0: "Background", 1: "Patella_L", 2: "Patella_R", 3: "Tibia_R", 4: "Tibia_L", 5: "Fibula_L",
        #                   6: "Femur_L", 7: "Femur_R", 8: "Fibula_R"}
        new_class_names = {0: "Background"
                           ,1 : "Femur_R"
                            ,2 : "Fibula_R"
                            ,3 : "Tibia_L"
                            ,4 : "Femur_L"
                            ,5 : "Patella_L"
                            ,6 : "Patella_R"
                            ,7 : "Fibula_L"
                            ,8 : "Tibia_R"}
        #sort the new_class_names
        new_class_names = dict(sorted(new_class_names.items()))
        update_nrrd_class_names(mask_path, new_class_names)

    multiclass_mask, labels = load_multiclass_mask(mask_path)
    # Print how many classes are in the masks array
    print(f"Classes in the mask: {len(np.unique(multiclass_mask))}")


    viewer_with_colored_classes(image_array, multiclass_mask, image_spacing, labels)
'''
'''if __name__ == "__main__":
    # Paths to image and masks
    case = '240088-2' #TODO - HEREEEEEEEE
    print(f"Processing case: {case}")
    image_path = f'/Users/samyakarzazielbachiri/Documents/SegmentationAI/scripts/output/images/{case}_images.nrrd'
    mask_path = f"/Users/samyakarzazielbachiri/Documents/SegmentationAI/scripts/output/multiclass_masks/{case}_multiclass_mask.nrrd"
    # Load image and mask
    image_array, image_spacing, _ = align_image(image_path, flip=False)

    p = 1

    if p == 1:
        new_class_names = {0: "Background", 2: "Radius_L", 1: "Humerus_L", 4: "Hand_L", 3: "Ulna_L"}
        #sort the new_class_names
        new_class_names = dict(sorted(new_class_names.items()))
        update_nrrd_class_names(mask_path, new_class_names)

    multiclass_mask, labels = load_multiclass_mask(mask_path)

    viewer_with_colored_classes(image_array, multiclass_mask, image_spacing, labels)
'''

'''
### Example main 2 - On loop
if __name__ == "__main__":
    # For all cases in the segmentai_dataset. Search for the images and masks in the data folder.
    folder = '/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/iiimg2'
    cases = [f.split('_')[0] for f in os.listdir(folder) if f.endswith('.nrrd')]
    for case in cases:
        print(f"Processing case: {case}")
        image_path = f'/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/iiimg2/{case}_images.nrrd'
        mask_path = f'/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/multiclass_masks/{case}_multiclass_mask.nrrd'

        # Load image and mask
        image_array, image_spacing, _ = align_image(image_path, flip=True)
        multiclass_mask = load_multiclass_mask(mask_path)

        # Visualize with colored classes
        #viewer_with_colored_classes(image_array, multiclass_mask, image_spacing)'''