import SimpleITK as sitk
import numpy as np
import napari

def load_multiclass_mask(mask_path):
    """
    Load a multiclass mask from a file.
    """
    mask = sitk.ReadImage(mask_path)
    return sitk.GetArrayFromImage(mask)

def align_image(image_path, flip=True):
    """
    Load the image and get its properties for visualization.
    """
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    image_spacing = image.GetSpacing()
    if flip:
        image_array = np.flip(image_array, axis=0)
        image = sitk.GetImageFromArray(image_array)
        #image_spacing = image.GetSpacing()
        image.SetSpacing(image_spacing)
        sitk.WriteImage(image, image_path)

    return image_array, np.array(image_spacing), image_spacing

def create_color_map(multiclass_mask):
    """
    Generate a unique color map for each class in the multiclass mask.
    """
    unique_classes = np.unique(multiclass_mask)
    color_map = {cls: np.random.rand(3,) for cls in unique_classes if cls != 0}  # Skip background class
    return color_map

def viewer_with_colored_classes(image_array, multiclass_mask, image_spacing):
    """
    Create separate Napari viewers for axial, coronal, and sagittal views with colored classes.
    """
    # Generate color map for the classes
    color_map = create_color_map(multiclass_mask)

    # Axial view
    axial_viewer = napari.Viewer(title="Axial View")
    axial_spacing = image_spacing[[2, 1, 0]]
    axial_viewer.add_image(image_array, name="Axial Image", colormap="gray", scale=axial_spacing)
    for cls, color in color_map.items():
        class_mask = (multiclass_mask == cls).astype(np.uint8)
        axial_viewer.add_labels(
            class_mask,
            name=f"Class {cls}",
            scale=axial_spacing,  # Swap the axes for coronal view
            opacity=0.5,
            colormap={1: color},  # Map the binary mask to the unique color
        )

    # Coronal view
    coronal_viewer = napari.Viewer(title="Coronal View")
    coronal_image = np.swapaxes(image_array, 0, 1)
    coronal_image = np.flip(coronal_image, axis=(0, 1))
    coronal_spacing = image_spacing[[1, 2, 0]]
    coronal_viewer.add_image(coronal_image, name="Coronal Image", colormap="gray", scale=coronal_spacing)
    for cls, color in color_map.items():
        class_mask = (multiclass_mask == cls).astype(np.uint8)
        coronal_mask = np.swapaxes(class_mask, 0, 1)
        coronal_mask = np.flip(coronal_mask, axis=(0, 1))
        coronal_viewer.add_labels(
            coronal_mask,
            name=f"Class {cls}",
            scale=coronal_spacing,
            opacity=0.5,
            colormap={1: color},
        )

    # Sagittal view
    sagittal_viewer = napari.Viewer(title="Sagittal View")
    sagittal_image = np.swapaxes(image_array, 0, 2)
    sagittal_image = np.rot90(sagittal_image, k=1, axes=(1, 2))
    sagittal_spacing = image_spacing[[1, 2, 0]]
    sagittal_viewer.add_image(sagittal_image, name="Sagittal Image", colormap="gray", scale=sagittal_spacing)
    for cls, color in color_map.items():
        class_mask = (multiclass_mask == cls).astype(np.uint8)
        sagittal_mask = np.swapaxes(class_mask, 0, 2)
        sagittal_mask = np.rot90(sagittal_mask, k=1, axes=(1, 2))
        sagittal_viewer.add_labels(
            sagittal_mask,
            name=f"Class {cls}",
            scale=sagittal_spacing,
            opacity=0.5,
            colormap={1: color},
        )

    # Run Napari viewers
    napari.run()

'''
### Example main 1 - On single case
if __name__ == "__main__":
    # Paths to image and masks
    case = '240042-1'

    image_path = f'/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/images/{case}_images.nrrd'
    mask_path = f'/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/multiclass_masks/{case}_multiclass_mask.nrrd'

    # Load image and mask
    image_array, image_spacing, _ = align_image(image_path, flip=False)
    multiclass_mask = load_multiclass_mask(mask_path)

    # Visualize with colored classes
    viewer_with_colored_classes(image_array, multiclass_mask, image_spacing)
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