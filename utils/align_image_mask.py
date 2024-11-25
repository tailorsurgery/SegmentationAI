import SimpleITK as sitk
import numpy as np
import napari

def align_mask_to_image(image_path, mask_path):
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

    return image_array, resampled_mask_array

def viewer(image_array, resampled_mask_array):
    # Prepare axial, coronal, and sagittal views
    # Axial view (original data)
    axial_image = image_array
    axial_mask = resampled_mask_array

    # Coronal view: swap axes 0 (slice axis) and 1 (height)
    coronal_image = np.swapaxes(image_array, 0, 1)
    coronal_mask = np.swapaxes(resampled_mask_array, 0, 1)

    # Sagittal view: swap axes 0 (slice axis) and 2 (width)
    sagittal_image = np.swapaxes(image_array, 0, 2)
    sagittal_mask = np.swapaxes(resampled_mask_array, 0, 2)

    # Start Napari viewer
    viewer = napari.Viewer()

    # Add axial view
    viewer.add_image(axial_image, name='Axial Image', colormap='gray')
    viewer.add_labels(axial_mask, name='Axial Mask')

    # Add coronal view
    viewer.add_image(coronal_image, name='Coronal Image', colormap='gray')
    viewer.add_labels(coronal_mask, name='Coronal Mask')

    # Add sagittal view
    viewer.add_image(sagittal_image, name='Sagittal Image', colormap='gray')
    viewer.add_labels(sagittal_mask, name='Sagittal Mask')

    # Organize layers into groups for clarity (optional)
    viewer.layers['Axial Image'].metadata = {'plane': 'axial'}
    viewer.layers['Axial Mask'].metadata = {'plane': 'axial'}
    viewer.layers['Coronal Image'].metadata = {'plane': 'coronal'}
    viewer.layers['Coronal Mask'].metadata = {'plane': 'coronal'}
    viewer.layers['Sagittal Image'].metadata = {'plane': 'sagittal'}
    viewer.layers['Sagittal Mask'].metadata = {'plane': 'sagittal'}

    # Run the Napari event loop
    napari.run()

# Paths to NRRD files
image_path = '/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/images/240093-2_images.nrrd'
mask_path = '/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/masks/240093-2_mask_2.nrrd'

image_array, resampled_mask_array = align_mask_to_image(image_path, mask_path)
viewer(image_array, resampled_mask_array)
