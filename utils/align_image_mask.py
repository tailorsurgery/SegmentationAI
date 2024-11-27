import SimpleITK as sitk
import numpy as np
import napari
import glob
import os

def align_mask_to_image(image_path, mask_path):
    """
    Align a single mask to the image space.
    """
    # Read the image and mask
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    # Resample the mask to the image's space
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_mask = resampler.Execute(mask)

    # Convert to NumPy arrays
    image_array = sitk.GetArrayFromImage(image)
    resampled_mask_array = sitk.GetArrayFromImage(resampled_mask)

    # Get spacing (voxel size)
    image_spacing = np.array(image.GetSpacing())

    return image_array, resampled_mask_array, image_spacing

def load_all_masks(image_path, case_prefix, mask_dir):
    """
    Load all masks with the given case prefix, align them to the image, 
    and store them in an array.
    """
    # Find all masks starting with the case prefix
    mask_paths = glob.glob(os.path.join(mask_dir, f"{case_prefix}_mask_*.nrrd"))
    if not mask_paths:
        print(f"No masks found for case: {case_prefix}")
        return None, []

    print(f"Masks found for case {case_prefix}: {mask_paths}")

    # Align each mask to the image
    aligned_masks = []
    for mask_path in mask_paths:
        _, resampled_mask_array, _ = align_mask_to_image(image_path, mask_path)
        aligned_masks.append(resampled_mask_array)

    # Stack masks into a single array (new dimension for each mask)
    stacked_masks = np.stack(aligned_masks, axis=0)
    return stacked_masks, mask_paths

def viewer_with_orientations(image_array, aligned_masks_array, mask_paths, image_spacing):
    """
    Create Napari viewers for axial, sagittal, coronal, and 3D views,
    allowing interposing of masks with the image.
    """
    # Axial view (ZYX)
    axial_viewer = napari.Viewer(title="Axial View")
    axial_viewer.add_image(image_array, name="Axial Image", colormap="gray", scale=image_spacing)

    for i, mask in enumerate(aligned_masks_array):
        axial_viewer.add_labels(mask, name=f"Axial Mask {os.path.basename(mask_paths[i])}", scale=image_spacing)

    # Coronal view (YXZ)
    coronal_viewer = napari.Viewer(title="Coronal View")
    coronal_image = np.swapaxes(image_array, 0, 1)
    coronal_image = np.flip(coronal_image, axis=(0, 1))
    
    coronal_masks = [np.swapaxes(mask, 0, 1) for mask in aligned_masks_array]
    coronal_masks = [np.flip(mask, axis=(0, 1)) for mask in coronal_masks]

    coronal_spacing = image_spacing[[1, 2, 0]]
    coronal_viewer.add_image(coronal_image, name="Coronal Image", colormap="gray", scale=coronal_spacing)
    for i, mask in enumerate(coronal_masks):
        coronal_viewer.add_labels(mask, name=f"Coronal Mask {os.path.basename(mask_paths[i])}", scale=coronal_spacing)

    # Sagittal view (XZY)
    sagittal_viewer = napari.Viewer(title="Sagittal View")

    sagittal_image = np.swapaxes(image_array, 0, 2)
    sagittal_image = np.rot90(sagittal_image, k=1, axes=(1, 2))

    sagittal_masks = [np.swapaxes(mask, 0, 2) for mask in aligned_masks_array]
    sagittal_masks = [np.rot90(mask, k=1, axes=(1, 2)) for mask in sagittal_masks]
    
    sagittal_spacing = image_spacing[[2, 1, 0]]
    sagittal_viewer.add_image(sagittal_image, name="Sagittal Image", colormap="gray", scale=sagittal_spacing)
    for i, mask in enumerate(sagittal_masks):
        sagittal_viewer.add_labels(mask, name=f"Sagittal Mask {os.path.basename(mask_paths[i])}", scale=sagittal_spacing)

    # 3D view
    '''viewer_3d = napari.Viewer(title="3D View")
    viewer_3d.add_image(image_array, name="3D Image", colormap="gray", scale=image_spacing, rendering="mip")
    for i, mask in enumerate(aligned_masks_array):
        viewer_3d.add_labels(mask, name=f"3D Mask {os.path.basename(mask_paths[i])}", scale=image_spacing, opacity=0.5)
    '''
    # Set 3D display mode for the 3D view
    '''viewer_3d.dims.ndisplay = 3'''

    # Run Napari viewers
    napari.run()


# Paths to image and masks
case = '240064'
image_path = f'/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/images/{case}_images.nrrd'
mask_dir = '/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/masks'

# Load image and aligned masks
aligned_masks_array, mask_paths = load_all_masks(image_path, case, mask_dir)

if aligned_masks_array is not None:
    # Get voxel spacing and visualize
    image_array, _, image_spacing = align_mask_to_image(image_path, mask_paths[0])
    viewer_with_orientations(image_array, aligned_masks_array, mask_paths, image_spacing)
else:
    print("No masks to visualize.")