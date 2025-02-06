import SimpleITK as sitk


def swap_xz_axes_in_mask(input_mask_path, output_mask_path):
    """
    Swap the X and Z axes of a 3D mask and save the new mask.

    Args:
        input_mask_path (str): Path to the original mask (3D).
        output_mask_path (str): Path to save the newly reordered mask.
    """
    # Read the original mask
    mask_image = sitk.ReadImage(input_mask_path)

    permute_filter = sitk.PermuteAxesImageFilter()
    permute_filter.SetOrder([2, 1, 0])

    # Apply the filter
    swapped_mask = permute_filter.Execute(mask_image)

    # Write out the reordered mask
    sitk.WriteImage(swapped_mask, output_mask_path)
    print(f"Saved mask with swapped X and Z axes to: {output_mask_path}")


if __name__ == "__main__":
    # Example usage
    case = "240066"
    region = "arms"
    original_mask = f"/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/multiclass_masks/{region}/{case}_multiclass_mask.nrrd"
    new_mask = f"/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/multiclass_masks/{region}/{case}_multiclass_mask.nrrd"
    swap_xz_axes_in_mask(original_mask, new_mask)