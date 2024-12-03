import gc
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

    images_option = int(input("Enter the number of the best images you want to keep: (0 to exit) "))
    count = 0
    while images_option != 0:  # Fix SyntaxWarning
        case = f'{case_number}-{images_option}'
        
        image_path = f'{output_dir}/images/{case}_images.nrrd'
        mask_path = f'{output_dir}/multiclass_masks/{case}_multiclass_mask.nrrd'
        print(f"Image path: {image_path}")
        print(f"Mask path: {mask_path}")

        try:
            # Load image
            image_array, image_spacing_array, image_spacing = align_image(image_path, flip=False)

            # TODO: Create multiclass mask
            # multiclass_mask = create_multiclass_mask(image_array)

            # Load multiclass mask
            multiclass_mask = load_multiclass_mask(mask_path)
            
            # Visualize with colored classes
            viewer_with_colored_classes(image_array, multiclass_mask, image_spacing_array)
            del image_array, image_spacing_array, image_spacing, multiclass_mask

        except Exception as e:
            print(f"Error processing case {case}: {e}")

        images_option = int(input("Are there more images you want to keep? Enter the number of images you want to keep: (0 to exit) "))
        
        count += 1