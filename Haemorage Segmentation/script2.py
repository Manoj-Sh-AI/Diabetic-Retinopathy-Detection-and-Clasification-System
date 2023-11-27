import os
import cv2

def augment_data(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

    for file in image_files:
        # Read the image
        image_path = os.path.join(input_folder, file)
        original_image = cv2.imread(image_path)

        # Flip horizontally
        flip_horizontal = cv2.flip(original_image, 1)
        cv2.imwrite(os.path.join(output_folder, f'{file[:-4]}_flip_horizontal.tif'), flip_horizontal)

        # Flip vertically
        flip_vertical = cv2.flip(original_image, 0)
        cv2.imwrite(os.path.join(output_folder, f'{file[:-4]}_flip_vertical.tif'), flip_vertical)

        # Flip both horizontally and vertically
        flip_both = cv2.flip(original_image, -1)
        cv2.imwrite(os.path.join(output_folder, f'{file[:-4]}_flip_both.tif'), flip_both)

if __name__ == "__main__":
    input_folder = "sample input/"  # Change this to your input folder path
    output_folder = "sample output/"  # Change this to your output folder path

    augment_data(input_folder, output_folder)
