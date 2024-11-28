from utils.dicom2nrrd import *

# Main function
if __name__ == "__main__":
    # Set up directories
    input_dir = '/Users/samyakarzazielbachiri/Documents/SegmentationAI/scripts/input'
    output_dir = '/Users/samyakarzazielbachiri/Documents/SegmentationAI/scripts/output'
    os.makedirs(output_dir, exist_ok=True)

    case_number = input("Enter the starting case number: ")
    count = 1

    # Process each DICOM study in the input folder
    for case in os.listdir(input_dir):
        case_path = os.path.join(input_dir, case)
        if not os.path.isdir(case_path):
            continue
        print(f"Processing the case")

        # Process all series within the case folder
        try:
            count = load_and_convert_dicom_to_nrrd(case_path, output_dir, case_number, count)
        except Exception as e:
            print(f"Failed to process case {case}: {e}")

    print("All cases processed successfully.")


    images_option = input("Enter the number of the best images you want to keep: ")
    img_path = f'/Users/samyakarzazielbachiri/Documents/SegmentationAI/scripts/output/{case_number}-{images_option}_images.nrrd'


