import numpy as np
import SimpleITK as sitk

def flip_image_and_mask_for_laterality(image_path, mask_path, save=False):
    """
    Flip the image and the mask to transform laterality (e.g., right knee to left knee).

    Parameters:
        image_path (str): Path to the image file.
        mask_path (str): Path to the mask file.
        save (bool): Whether to save the flipped files back to disk.

    Returns:
        flipped_image_array (numpy.ndarray): Flipped image array.
        flipped_mask_array (numpy.ndarray): Flipped mask array.
    """
    # Load the image and mask
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    # Get the image and mask as numpy arrays
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    # Flip along the left-right axis (axis=2 for coronal view)
    flipped_image_array = np.flip(image_array, axis=2)
    flipped_mask_array = np.flip(mask_array, axis=2)

    # Convert flipped arrays back to SimpleITK images
    flipped_image = sitk.GetImageFromArray(flipped_image_array)
    flipped_mask = sitk.GetImageFromArray(flipped_mask_array)

    # Preserve spacing and origin
    flipped_image.SetSpacing(image.GetSpacing())
    flipped_image.SetOrigin(image.GetOrigin())
    flipped_image.SetDirection(image.GetDirection())

    flipped_mask.SetSpacing(mask.GetSpacing())
    flipped_mask.SetOrigin(mask.GetOrigin())
    flipped_mask.SetDirection(mask.GetDirection())

    if save:
        # Save the flipped images
        sitk.WriteImage(flipped_image, image_path)
        sitk.WriteImage(flipped_mask, mask_path)
        print(f"Flipped image saved to: {image_path}")
        print(f"Flipped mask saved to: {mask_path}")

    return flipped_image_array, flipped_mask_array

if __name__ == "__main__":
    case = "240093-2-2"
    path = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset"
    image_path = path + f"/images/leg/knee/{case}_images.nrrd"
    mask_path = path + f"/multiclass_masks/leg/knee/{case}_multiclass_mask.nrrd"
    flip_image_and_mask_for_laterality(image_path, mask_path, save=True)
