import numpy as np
import SimpleITK as sitk

def flip_image_and_mask_for_laterality(image_path, save=False):
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
    #mask = sitk.ReadImage(mask_path)

    # Get the image and mask as numpy arrays
    image_array = sitk.GetArrayFromImage(image)
    #mask_array = sitk.GetArrayFromImage(mask)

    # Flip along the left-right axis (axis=2 for coronal view)
    flipped_image_array = np.flip(image_array, axis=2)
    #flipped_mask_array = np.flip(mask_array, axis=2)

    # Convert flipped arrays back to SimpleITK images
    flipped_image = sitk.GetImageFromArray(flipped_image_array)
    #flipped_mask = sitk.GetImageFromArray(flipped_mask_array)

    # Preserve spacing and origin
    flipped_image.SetSpacing(image.GetSpacing())
    flipped_image.SetOrigin(image.GetOrigin())
    flipped_image.SetDirection(image.GetDirection())

    """flipped_mask.SetSpacing(mask.GetSpacing())
    flipped_mask.SetOrigin(mask.GetOrigin())
    flipped_mask.SetDirection(mask.GetDirection())"""

    if save:
        # Save the flipped images
        sitk.WriteImage(flipped_image, image_path)
        #sitk.WriteImage(flipped_mask, mask_path)
        print(f"Flipped image saved to: {image_path}")


    return flipped_image_array

if __name__ == "__main__":
    """for case in ["240023-2", "240025-2", "240031-1", "240031-2", "240033", "240042-1", "240043-1",
                 "240045", "240046", "240047", "240050-1", "240050-2", "240057", "240066", "240069-1",
                 "240069-2", "240088-1", "240088-2"]:"""
    case = "240066"
    path = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset"
    image_path = path + f"/images/arms/{case}_images.nrrd"

    flip_image_and_mask_for_laterality(image_path, save=True)
