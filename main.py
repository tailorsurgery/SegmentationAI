from utils.dicom2nrrd import *
from utils.align_image_multiclass_mask import *


# Main function
if __name__ == "__main__":
    # Set up directories
    input_dir = '/Users/samyakarzazielbachiri/Documents/SegmentationAI/scripts/input'
    output_dir = '/Users/samyakarzazielbachiri/Documents/SegmentationAI/scripts/output'
    os.makedirs(output_dir, exist_ok=True)

    case_number = input("Enter the starting case number: ")
    process_images(input_dir, f'{output_dir}/images', case_number)

    images_option = input("Enter the number of the best images you want to keep: ")
    case = f'{case_number}-{images_option}'
    
    image_path = f'{output_dir}/images/{case}_images.nrrd'
    mask_path = f'{output_dir}/multiclass_masks/{case}_multiclass_mask.nrrd'

    # Load image
    image_array, image_spacing_array, image_spacing = align_image(image_path, flip=False)
    

    # TODO: Create multiclass mask

    # Load multiclass mask
    multiclass_mask = load_multiclass_mask(mask_path)
    
    
    # Visualize with colored classes
    viewer_with_colored_classes(image_array, multiclass_mask, image_spacing_array)

